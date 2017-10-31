"""
MNIST-specific APIs.
"""

# pylint: disable=E1129

import tensorflow as tf

def fully_connected_classifier(input_ph):
    """
    Apply a simple fully-connected network to the image to
    produce output logits.
    """
    hidden = tf.contrib.layers.fully_connected(input_ph, 300)
    return tf.contrib.layers.fully_connected(hidden, 10, activation_fn=None)

def conv_classifier(input_ph):
    """
    Apply a simple convolutional network to the image to
    produce output logits.
    """
    shaped_in = tf.reshape(input_ph, [-1, 28, 28, 1])
    with tf.variable_scope('conv_1'):
        out_1 = tf.layers.conv2d(shaped_in, 64, 5, strides=[2, 2], activation=tf.nn.relu)
    with tf.variable_scope('conv_2'):
        out_2 = tf.layers.conv2d(out_1, 64, 5, activation=tf.nn.relu)
    out_size = 1
    for dim in out_2.get_shape()[1:]:
        out_size *= int(dim)
    flat_out = tf.reshape(out_2, [tf.shape(out_2)[0], out_size])
    return tf.contrib.layers.fully_connected(flat_out, 10, activation_fn=None)
