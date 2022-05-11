# tensorflow should upgrade to 2.8.0
from distutils.version import LooseVersion
import datetime
from collections import OrderedDict
from collections import namedtuple

import numpy as np
import numpy.random as rd
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.framework.ops import Tensor

if LooseVersion(tf.__version__) >= LooseVersion("2.8"):
    from tensorflow.python.ops.variables import Variable, RefVariable
else:
    print("Using tensorflow version older then 2.8.0 -> skipping RefVariable storing")
    from tensorflow.python.ops.variables import Variable

from lsnn.toolbox.rewiring_tools import weight_sampler
from lsnn.toolbox.tensorflow_einsums.einsum_re_written import einsum_bi_ijk_to_bjk
from lsnn.toolbox.tensorflow_utils import tf_roll

from time import time

Cell = tf.contrib.rnn.BasicRNNCell


def placeholder_container_for_rnn_state(cell_state_size, dtype, batch_size, name='TupleStateHolder'):
    '''Create a tuple as container for RNN states.
    Input Args:
        cell_state_size : a named tuple
        dtype : data type of the states
        batch_size : int
    Returns:
        placeholder_tuple : a tuple to hold places for RNN states
    '''
    with tf.name_scope(name):
        default_dict = cell_state_size._asdict() # convert the named tuple 'cell_state_size' into a python dict instance
        placeholder_dict = OrderedDict({}) # create a OrderedDict which is a python dict that preserves the order in which the keys are inserted. 
        for k, v in default_dict.items(): # k - key, v - value
            # if v is not a list, convert that v to a list
            if np.shape(v) == (): 
                v = [v]
            shape = np.concatenate([[batch_size], v]) # the shape of each layer?
            placeholder_dict[k] = tf.placeholder(shape=shape, dtype=dtype, name=k)
        # **d means "treat the key-value pairs in the dictionary as additional named arguments to this function call."
        # def foo(x, y):
        #     print(x, y)
        #>>> d = {'x':1, 'y':2}
        # >>> foo(**d)
        # 1 2
        placeholder_tuple = cell_state_size.__class__(**placeholder_dict)
        return placeholder_tuple


def feed_dict_with_placeholder_container(dict_to_update, state_holder, state_value, batch_selection=None):
    '''Update the dictionary 'dict_to_update'
    Input Args:
        dict_to_update : the dictonary to be updated
    Returns:
        dict_to_update : the updated dictonary
    '''
    if state_value is None:
        return dict_to_update

    assert state_holder.__class__ == state_value.__class__, 'Should have the same class, got {} and {}'.format(
        state_holder.__class__, state_value.__class__)

    for k, v in state_value._asdict().items():
        if batch_selection is None:
            dict_to_update.update({state_holder._asdict()[k]: v}) # convert the state_holder as a dict and create the k-v pair
        else:
            dict_to_update.update({state_holder._asdict()[k]: v[batch_selection]})

    return dict_to_update


#################################
# Spike function
#################################

@tf.custom_gradient # Decorator to define a function with a custom gradient.

# STDP should be inserted in this function!!!!!
def SpikeFunction(v_scaled, dampening_factor):
    z_ = tf.math.greater(v_scaled, 0.) # returns the truth/flase value of (x > y) element-wise.
    z_ = tf.cast(z_, dtype=tf.float32) # cast z to data type of float32, i.e., true - 1.0, false - 0.0

    def grad(dy): 
        # calculate the gradient for BPTT
        dE_dz = dy # E - error
        dz_dv_scaled = tf.math.maximum(1 - tf.abs(v_scaled), 0)
        dz_dv_scaled *= dampening_factor # dampening_factor = 0.3 for sequantial MNIST in NIPS 2018 

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled,
                tf.zeros_like(dampening_factor)] 
        # tf.zeros_like(dampening_factor) - a tensor with shape of 'dampening_factor'.
    
    # STDP should be inserted here!!!!!

    # tf.identity : Return a Tensor with the same shape and contents as input.
    return tf.identity(z_, name="SpikeFunction"), grad


def weight_matrix_with_delay_dimension(w, d, n_delay):
    """
    Generate the tensor of shape n_in x n_out x n_delay that represents the synaptic weights with the right delays.
    Use for e-prop as an alternative of BPTT
    :param w: synaptic weight value, float tensor of shape (n_in x n_out)
    :param d: delay number, int tensor of shape (n_in x n_out)
    :param n_delay: number of possible delays
    :return: WD
    """
    with tf.name_scope('WeightDelayer'): 
        w_d_list = []
        for kd in range(n_delay):
            mask = tf.math.equal(d, kd) # Returns the truth value of (d == kd) element-wise.
            w_d = tf.where(condition=mask, x=w, y=tf.zeros_like(w)) # Returns the indices that meets the 'mask' condition.
            w_d_list.append(w_d) 

        delay_axis = len(d.shape)
        WD = tf.stack(w_d_list, axis=delay_axis)

    return WD


# PSP on output layer - probably used in alternative BPTT paper
def exp_convolve(tensor, decay):  # tensor shape (trial, time, neuron)
    with tf.name_scope('ExpConvolve'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float64]
        # tensor here is a 3D tensor, perm is used to specify the way of transpose
        tensor_time_major = tf.transpose(tensor, perm=[1, 0, 2]) 
        initializer = tf.zeros_like(tensor_time_major[0])
        # tf. scan - scan on the list of tensors unpacked from elems of 'tensor_time_major' on dimension 0.
        filtered_tensor = tf.scan(lambda a, x: a * decay + (1 - decay) * x, tensor_time_major, initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor, perm=[1, 0, 2])
    return filtered_tensor


LIFStateTuple = namedtuple('LIFStateTuple', ('v', 'z', 'i_future_buffer', 'z_buffer'))


def tf_cell_to_savable_dict(cell, sess, supplement={}):
    """
    Usefull function to return a python/numpy object from of of the tensorflow cell object defined here.
    The idea is simply that varaibles and Tensors given as attributes of the object with be replaced by there numpy value evaluated on the current tensorflow session.
    :param cell: tensorflow cell object
    :param sess: tensorflow session
    :param supplement: some possible
    :return:
    """

    dict_to_save = {}
    dict_to_save['cell_type'] = str(cell.__class__)
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dict_to_save['time_stamp'] = time_stamp

    dict_to_save.update(supplement)

    tftypes = [Variable, Tensor]
    if LooseVersion(tf.__version__) >= LooseVersion("2.8"):
        tftypes.append(RefVariable)

    for k, v in cell.__dict__.items():
        if k == 'self':
            pass
        elif type(v) in tftypes:
            dict_to_save[k] = sess.run(v)
        elif type(v) in [bool, int, float, np.int64, np.ndarray]:
            dict_to_save[k] = v
        else:
            print('WARNING: attribute of key {} and value {} has type {}, recoding it as string.'.format(k, v, type(v)))
            dict_to_save[k] = str(v)

    return dict_to_save


class LIF(Cell):
    def __init__(self, n_in, n_rec, tau=20., thr=0.03,
                 dt=1., n_refractory=0, dtype=tf.float32, n_delay=1, rewiring_connectivity=-1,
                 in_neuron_sign=None, rec_neuron_sign=None,
                 dampening_factor=0.3,
                 injected_noise_current=0.,
                 V0=1.):
        """
        Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.
        :param n_in: number of input neurons
        :param n_rec: number of recurrent neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step of the simulation
        :param n_refractory: number of refractory time steps
        :param dtype: data type of the cell tensors
        :param n_delay: number of synaptic delay timestep, the delay range goes from 1 to n_delay time steps
        :param reset: method of resetting membrane potential after spike thr-> by fixed threshold amount, zero-> to zero
        """

        if np.isscalar(tau): tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
        if np.isscalar(thr): thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)
        tau = tf.cast(tau, dtype=dtype)
        dt = tf.cast(dt, dtype=dtype)

        self.dampening_factor = dampening_factor

        # Parameters
        self.n_delay = n_delay
        self.n_refractory = n_refractory

        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.tau = tf.Variable(tau, dtype=dtype, name="Tau", trainable=False) # trainable=False, set tau as a constant
        self._decay = tf.exp(-dt / tau) # the alpha value in membrane voltage update equation
        self.thr = tf.Variable(thr, dtype=dtype, name="Threshold", trainable=False) 

        self.V0 = V0 # initial membrane voltage
        self.injected_noise_current = injected_noise_current # add some random noises

        self.rewiring_connectivity = rewiring_connectivity 
        self.in_neuron_sign = in_neuron_sign # input current from former layer
        self.rec_neuron_sign = rec_neuron_sign # recurrent current from recurrent neurons of current layer

        with tf.variable_scope('InputWeights'):

            # Input weights
            if 0 < rewiring_connectivity < 1:
                # weight_sampler from rewiring tool
                self.w_in_val, self.w_in_sign, self.w_in_var, _ = weight_sampler(n_in, n_rec, rewiring_connectivity,
                                                                                 neuron_sign=in_neuron_sign) 
            else:
                self.w_in_var = tf.Variable(rd.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype, name="InputWeight")
                self.w_in_val = self.w_in_var

            self.w_in_val = self.V0 * self.w_in_val
            self.w_in_delay = tf.Variable(rd.randint(self.n_delay, size=n_in * n_rec).reshape(n_in, n_rec),
                                          dtype=tf.int64, name="InDelays", trainable=False)
            self.W_in = weight_matrix_with_delay_dimension(self.w_in_val, self.w_in_delay, self.n_delay)

        with tf.variable_scope('RecWeights'):
            if 0 < rewiring_connectivity < 1:
                self.w_rec_val, self.w_rec_sign, self.w_rec_var, _ = weight_sampler(n_rec, n_rec,
                                                                                    rewiring_connectivity,
                                                                                    neuron_sign=rec_neuron_sign)
            else:
                if rec_neuron_sign is not None or in_neuron_sign is not None:
                    raise NotImplementedError('Neuron sign requested but this is only implemented with rewiring')
                self.w_rec_var = Variable(rd.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype,
                                          name='RecurrentWeight')
                self.w_rec_val = self.w_rec_var

            recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))

            self.w_rec_val = self.w_rec_val * self.V0
            self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val),
                                      self.w_rec_val)  # Disconnect autotapse
            self.w_rec_delay = tf.Variable(rd.randint(self.n_delay, size=n_rec * n_rec).reshape(n_rec, n_rec),
                                           dtype=tf.int64, name="RecDelays", trainable=False)
            self.W_rec = weight_matrix_with_delay_dimension(self.w_rec_val, self.w_rec_delay, self.n_delay)

    @property
    def state_size(self):
        return LIFStateTuple(v=self.n_rec,
                             z=self.n_rec,
                             i_future_buffer=(self.n_rec, self.n_delay),
                             z_buffer=(self.n_rec, self.n_refractory))

    @property
    def output_size(self):
        return self.n_rec

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        i_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_delay), dtype=dtype)
        z_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_refractory), dtype=dtype)

        return LIFStateTuple(
            v=v0,
            z=z0,
            i_future_buffer=i_buff0,
            z_buffer=z_buff0
        )

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):

        i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs, self.W_in) + einsum_bi_ijk_to_bjk(
            state.z, self.W_rec)

        new_v, new_z = self.LIF_dynamic(
            v=state.v,
            z=state.z,
            z_buffer=state.z_buffer,
            i_future_buffer=i_future_buffer)

        new_z_buffer = tf_roll(state.z_buffer, new_z, axis=2)
        new_i_future_buffer = tf_roll(i_future_buffer, axis=2)

        new_state = LIFStateTuple(v=new_v,
                                  z=new_z,
                                  i_future_buffer=new_i_future_buffer,
                                  z_buffer=new_z_buffer)
        return new_z, new_state

    def LIF_dynamic(self, v, z, z_buffer, i_future_buffer, thr=None, decay=None, n_refractory=None, add_current=0.):
        """
        Function that generate the next spike and voltage tensor for given cell state.
        :param v
        :param z
        :param z_buffer:
        :param i_future_buffer:
        :param thr:
        :param decay:
        :param n_refractory:
        :param add_current:
        :return:
        """

        if self.injected_noise_current > 0:
            add_current = tf.random_normal(shape=z.shape, stddev=self.injected_noise_current) # add random noise to current

        with tf.name_scope('LIFdynamic'):
            if thr is None: thr = self.thr
            if decay is None: decay = self._decay
            if n_refractory is None: n_refractory = self.n_refractory

            i_t = i_future_buffer[:, :, 0] + add_current

            I_reset = z * thr * self.dt # thr is fixed for LIF neuron, but changable for ALIF neuron. 

            new_v = decay * v + (1 - decay) * i_t - I_reset # the membrane voltage at t+dt

            # Spike generation
            v_scaled = (v - thr) / thr

            # new_z = differentiable_spikes(v_scaled=v_scaled)
            new_z = SpikeFunction(v_scaled, self.dampening_factor) # update the z value
            # return tf.identity(z_, name="SpikeFunction"), grad

            if n_refractory > 0:
                is_ref = tf.greater(tf.reduce_max(z_buffer[:, :, -n_refractory:], axis=2), 0)
                new_z = tf.where(is_ref, tf.zeros_like(new_z), new_z)

            new_z = new_z * 1 / self.dt

            return new_v, new_z # return the new membrane voltage, and new input spike train


ALIFStateTuple = namedtuple('ALIFState', (
    'z',
    'v',
    'b',

    'i_future_buffer',
    'z_buffer'))


class ALIF(LIF):
    def __init__(self, n_in, n_rec, tau=20, thr=0.01,
                 dt=1., n_refractory=0, dtype=tf.float32, n_delay=1,
                 tau_adaptation=200., beta=1.6,
                 rewiring_connectivity=-1, dampening_factor=0.3,
                 in_neuron_sign=None, rec_neuron_sign=None, injected_noise_current=0.,
                 V0=1.):
        """
        Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.
        :param n_in: number of input neurons
        :param n_rec: number of recurrent neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step of the simulation
        :param n_refractory: number of refractory time steps
        :param dtype: data type of the cell tensors
        :param n_delay: number of synaptic delay, the delay range goes from 1 to n_delay time steps
        :param tau_adaptation: adaptation time constant for the threshold voltage
        :param beta: amplitude of adpatation
        :param rewiring_connectivity: number of non-zero synapses in weight matrices (at initialization)
        :param in_neuron_sign: vector of +1, -1 to specify input neuron signs
        :param rec_neuron_sign: same of recurrent neurons
        :param injected_noise_current: amplitude of current noise
        :param V0: to choose voltage unit, specify the value of V0=1 Volt in the desired unit (example V0=1000 to set voltage in millivolts)
        """

        super(ALIF, self).__init__(n_in=n_in, n_rec=n_rec, tau=tau, thr=thr, dt=dt, n_refractory=n_refractory,
                                   dtype=dtype, n_delay=n_delay,
                                   rewiring_connectivity=rewiring_connectivity,
                                   dampening_factor=dampening_factor, in_neuron_sign=in_neuron_sign,
                                   rec_neuron_sign=rec_neuron_sign,
                                   injected_noise_current=injected_noise_current,
                                   V0=V0)

        if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None: raise ValueError("beta parameter for adaptive bias must be set")

        self.tau_adaptation = tf.Variable(tau_adaptation, dtype=dtype, name="TauAdaptation", trainable=False)

        self.beta = tf.Variable(beta, dtype=dtype, name="Beta", trainable=False)
        self.decay_b = np.exp(-dt / tau_adaptation) #pj in the equation

    @property
    def output_size(self):
        return [self.n_rec, self.n_rec, self.n_rec]

    @property
    def state_size(self):
        return ALIFStateTuple(v=self.n_rec,
                              z=self.n_rec,
                              b=self.n_rec,
                              i_future_buffer=(self.n_rec, self.n_delay),
                              z_buffer=(self.n_rec, self.n_refractory))

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        b0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        i_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_delay), dtype=dtype)
        z_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_refractory), dtype=dtype)

        return ALIFStateTuple(
            v=v0,
            z=z0,
            b=b0,
            i_future_buffer=i_buff0,
            z_buffer=z_buff0
        )

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        with tf.name_scope('ALIFcall'):
            i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs, self.W_in) + einsum_bi_ijk_to_bjk(
                state.z, self.W_rec)

            new_b = self.decay_b * state.b + (1. - self.decay_b) * state.z # equation of bj(t+delta_t)

            thr = self.thr + new_b * self.beta * self.V0 # the threshold voltage is changable for ALIF neuron

            new_v, new_z = self.LIF_dynamic(
                v=state.v,
                z=state.z,
                z_buffer=state.z_buffer,
                i_future_buffer=i_future_buffer,
                decay=self._decay,
                thr=thr)

            new_z_buffer = tf_roll(state.z_buffer, new_z, axis=2)
            new_i_future_buffer = tf_roll(i_future_buffer, axis=2)

            new_state = ALIFStateTuple(v=new_v,
                                       z=new_z,
                                       b=new_b,
                                       i_future_buffer=new_i_future_buffer,
                                       z_buffer=new_z_buffer)
        return [new_z, new_v, thr], new_state


def static_rnn_with_gradient(cell, inputs, state, loss_function, T, verbose=True):
    batch_size = tf.shape(inputs)[0]
    # create empty lists to store values of RNN network
    thr_list = []
    state_list = []
    z_list = []
    v_list = []

    if verbose: print('Building forward Graph...', end=' ')
    t0 = time()
    for t in range(T): # T -- total time steps
        outputs, state = cell(inputs[:, t, :], state)
        z, v, thr = outputs
        # append values to the created lists
        z_list.append(z)
        v_list.append(v)
        thr_list.append(thr)
        state_list.append(state)
    # tf.stack -- Stacks a list of rank-R tensors into one rank-(R+1) tensor.
    zs = tf.stack(z_list, axis=1)
    vs = tf.stack(v_list, axis=1)
    thrs = tf.stack(thr_list, axis=1)
    loss = loss_function(zs)

    de_dz_partial = tf.gradients(loss, zs)[0]
    if de_dz_partial is None:
        de_dz_partial = tf.zeros_like(zs)
        print('Warning: Partial de_dz is None')
    print('Done in {:.2f}s'.format(time() - t0))

    def namedtuple_to_list(state):
        return list(state._asdict().values())

    zero_state_as_list = cell.zero_state(batch_size, tf.float32)
    de_dstate = namedtuple_to_list(cell.zero_state(batch_size, dtype=tf.float32))
    g_list = []
    if verbose: print('Building backward Graph...', end=' ')
    t0 = time()
    for t in np.arange(T)[::-1]:

        # gradient from next state
        if t < T - 1:
            state = namedtuple_to_list(state_list[t])
            next_state = namedtuple_to_list(state_list[t + 1])
            # tf.gradients -- Constructs symbolic derivatives of sum of ys w.r.t. x in xs
            de_dstate = tf.gradients(ys=next_state, xs=state, grad_ys=de_dstate)

            for k_var, de_dvar in enumerate(de_dstate):
                if de_dvar is None:
                    de_dstate[k_var] = tf.zeros_like(zero_state_as_list[k_var])
                    print('Warning: var {} at time {} is None'.format(k_var, t))

        # add the partial derivative due to current error
        de_dstate[0] = de_dstate[0] + de_dz_partial[:, t]
        g_list.append(de_dstate[0])

    g_list = list(reversed(g_list))

    gs = tf.stack(g_list, axis=1)
    print('Done in {:.2f}s'.format(time() - t0))

    return zs, vs, thrs, gs, state_list[-1]