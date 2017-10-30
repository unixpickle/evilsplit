"""
MNIST-specific APIs.
"""

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

_MNIST_IMAGES = None
_MNIST_LABELS = None

def get_data(indices):
    """
    Get (inputs, outputs) for the training samples.
    """
    # pylint: disable=W0603
    global _MNIST_IMAGES
    global _MNIST_LABELS
    if _MNIST_LABELS is None:
        data = input_data.read_data_sets('MNIST_data', one_hot=True)
        _MNIST_LABELS = np.array(list(data.train.labels) + list(data.validation.labels) +
                                 list(data.test.labels))
        _MNIST_IMAGES = np.array(list(data.train.images) + list(data.validation.images) +
                                 list(data.test.images))
    return _MNIST_IMAGES[indices], _MNIST_LABELS[indices]

def sample_classes(indices):
    """
    Get the classes for the sample indices.

    Each class is an integer.
    """
    return np.argmax(get_data(indices)[1], axis=-1)

def sample_images(indices):
    """
    Get the images for the sample indices.
    """
    return get_data(indices)[0]
