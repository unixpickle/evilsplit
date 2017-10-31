"""
Algorithms for creating adversarial splits.
"""

import numpy as np

from .mnist import sample_classes
from .models import fully_connected_classifier
from .train import train_and_test

def min_accuracy_split(architecture=fully_connected_classifier):
    """
    Create a split by training an architecture and
    selecting the least-fit samples.

    Returns (train_indices, test_indices)
    """
    all_indices = np.arange(70000)
    _, all_losses = train_and_test(all_indices, all_indices, architecture=architecture)
    best_to_worst = np.array(next(zip(*sorted(enumerate(all_losses), key=lambda x: x[1]))))
    return _even_adversarial_split(best_to_worst)

def _even_adversarial_split(best_to_worst):
    """
    Select the 1000 worst samples from each class and turn
    that into the test set.

    Returns (train, test).
    """
    labels = sample_classes(best_to_worst)
    train, test = [], []
    for label in range(10):
        samples = best_to_worst[labels == label]
        train.extend(samples[:-1000])
        test.extend(samples[-1000:])
    return train, test
