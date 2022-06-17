import tensorflow as tf
import numpy as np
import time 

# (TODO) - Step 1: define some constants

# Constants for the model
input_n = tf.constant(80)
input_exitatory_n = tf.constant(60)
input_inhibitary_n = tf.constant(20)
hidden_n = tf.constant(220)
hidden_LIF_n = tf.constant(120)
hidden_ALIF_n = hidden_n - hidden_LIF_n
output_n = tf.constant(10)

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
# (TODO) - Step 3: define the model
# (TODO) - Step 4: fit the model
# (TODO) - Step 5: test and calculate the accuracy  