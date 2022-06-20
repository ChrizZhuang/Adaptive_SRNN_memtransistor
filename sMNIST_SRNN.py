import string
import tensorflow as tf
import numpy as np
import time 
import pickle, gzip, os, sys, signal

# (TODO) - Step 1: define some constants

# Constants for the model
input_n = tf.constant(80)
input_exitatory_n = tf.constant(60)
input_inhibitary_n = tf.constant(20)
hidden_n = tf.constant(220)
hidden_LIF_n = tf.constant(120)
hidden_ALIF_n = hidden_n - hidden_LIF_n
output_n = tf.constant(10)

# Constants for training and testing
print("Please specify the constants for STDP weight update.")

training_size = tf.constant(int(input(" Training size: ")))
test_size = tf.constant(int(input(" Test size: ")))

# Constants for STDP
print("Please specify the constants for STDP weight update.")

# potentiation
alpha_p = tf.constant(float(input(" alpha_p: ")))
beta_p = tf.constant(float(input(" beta_p: ")))

# depression
alpha_m = tf.constant(float(input(" alpha_m: ")))
beta_m = tf.constant(float(input(" beta_m: ")))

# device parameters
G_max = tf.constant(float(input(" G_max: ")))
G_min = tf.constant(float(input(" G_min: ")))

t0 = 0

# (TODO) - Step 2: read in the MNIST images and convert them to spike trains

def load_training_set(mnist_directory, training_size):
    """
    Load the MNIST digits for training. Shuffles them as well.

    Inputs: 
        mnist_directory - string: the directory to store MNIST dataset
        training_size - int : the size of training size
    Return:
        training_set - np.array: training_set of randomly selected images from MNIST with size of training_size
    """
    # Sanity checks
    assert(type(mnist_directory) == string), "Parameter mnist_directory must be a string!"
    assert(type(training_size) == int), "Parameter training_size must be an int!"
    assert(training_size >=0 and training_size <= 60000), "Parameter training_size must >= 0 and <= 60000!"

    print("Loading MNIST for training (randomized)...")
    
    # Update global training_set variable
    global training_set
    
    # Open MNIST pickle package
    f = gzip.open(mnist_directory, 'rb')
    
    # load sets using pickle
    # encoding necessary since pickle file was created in python 2.7
    train, valid, _ = pickle.load(f,encoding='latin1')
    
    # split up into corresponding sets
    [training_set, _] = train
    [validation_set, _] = valid
    
    # update training set
    training_set = np.concatenate((training_set, validation_set))
    
    # close MNIST file
    f.close()
    
    # randomize loaded MNIST set
    np.random.shuffle(training_set)

    #random_index = np.random.choice(training_set)
    random_index = np.random.choice(training_set.shape[0],training_size,replace=False)
    training_set = training_set[random_index]
    
    print("Done!")

    return training_set


# Reference repo: 
# https://github.com/intel-nrc-ecosystem/models/blob/master/nxsdk_modules_ncl/lsnn/tutorials/smnist_tutorial.ipynb
# Part: Encoding the images into spikes
# spike encoding example
def find_onset_offset(y, threshold):
    """
    Given the input signal y with samples,
    find the indices where y increases and descreases through the value threshold.
    Return stacked binary arrays of shape y indicating onset and offset threshold crossings.
    y must be 1-D numpy arrays.
    """
    # Sanity check
    assert (len(np.array(threshold)) == 1), "The length of threshold should be 1!"

    if threshold == 255:
        equal = y == threshold
        transition_touch = np.where(equal)[0]
        touch_spikes = np.zeros_like(y)
        touch_spikes[transition_touch] = 1
        return np.expand_dims(touch_spikes, axis=0)
    else:
        # Find where y crosses the threshold (increasing).
        lower = y < threshold
        higher = y >= threshold
        transition_onset = np.where(lower[:-1] & higher[1:])[0]
        transition_offset = np.where(higher[:-1] & lower[1:])[0]
        onset_spikes = np.zeros_like(y)
        offset_spikes = np.zeros_like(y)
        onset_spikes[transition_onset] = 1
        offset_spikes[transition_offset] = 1

    return np.stack((onset_spikes, offset_spikes))


def generate_spike_train_from_image(image, n_inputs = 80):
    """Generate spike trains from an given image.
    Input:
        image: a numpy array that represent pixels of an image
        n_inputs: number of input neurons.
        n_thresholds: number of thresholds for the input layer.
                      should be around half of the number of input neurons. 
    Output
        spike_train_image: spike train of a given image
    """
    # Sanity check
    assert(type(image) == np.array), "Parameter image should be a np.array!"
    assert(type(n_inputs) == int), "Parameter n_inputs should be an int!"

    # turn the image into a 1D array
    image = image.reshape(-1, 1)

    thresholds = np.linspace(0, 255, n_inputs // 2)

    spike_train_image = []
    for pixel in image:  # shape img = (784)
        Sspikes = None
        for thr in thresholds:
            if Sspikes is not None:
                Sspikes = np.concatenate((Sspikes, find_onset_offset(pixel, thr)))
            else:
                Sspikes = find_onset_offset(pixel, thr)
        Sspikes = np.array(Sspikes)  # shape Sspikes = (31, 784)
        Sspikes = np.swapaxes(Sspikes, 0, 1)
        spike_train_image.append(Sspikes)
    spike_train_image = np.array(spike_train_image)
    # add output cue neuron, and expand time for two image rows (2*28)
    out_cue_duration = 2*28
    spike_train_image = np.lib.pad(spike_train_image, ((0, 0), (0, out_cue_duration), (0, 1)), 'constant')
    # output cue neuron fires constantly for these additional recall steps
    spike_train_image[:, -out_cue_duration:, -1] = 1

    return spike_train_image

def image_process_pipeline(mnist_directory, training_size):
    training_set = load_training_set(mnist_directory, training_size)
    spike_trains_training_set = []
    for image in training_set:
        spike_trains_training_set.append(generate_spike_train_from_image(image))

    return np.array(spike_trains_training_set)

# (TODO) - Step 3: define the model
# (TODO) - Step 4: fit the model
# (TODO) - Step 5: test and calculate the accuracy  