"""
Train on all MNIST digits and then select the worst ones
to be used as the test set.

Measure accuracy on the results.
"""

import numpy as np

from evilsplit import train_and_test

NUM_SAMPLES = 65000

def main():
    """
    Program entry-point.
    """
    print('Training on all data...')
    _, all_losses = train_and_test(np.arange(NUM_SAMPLES), np.arange(NUM_SAMPLES))
    best_to_worst = np.array(next(zip(*sorted(enumerate(all_losses), key=lambda x: x[1]))))
    train_indices = best_to_worst[:-10000]
    test_indices = best_to_worst[-10000:]
    print('Training on adversarial split.')
    corrects, _ = train_and_test(train_indices, test_indices)
    print('Mean accuracy: ' + str(np.mean(np.array(corrects).astype('float32'))))

if __name__ == '__main__':
    main()
