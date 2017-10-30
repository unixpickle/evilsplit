"""
MNIST-specific APIs.
"""

import tensorflow as tf

def fully_connected_classifier(input_ph):
    """
    Apply a simple fully-connected network to the image to
    produce output logits.
    """
    hidden = tf.contrib.layers.fully_connected(input_ph, 300)
    return tf.contrib.layers.fully_connected(hidden, 10, activation_fn=None)
