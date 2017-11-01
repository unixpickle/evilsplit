"""
Train on all MNIST digits and then select the worst ones
to be used as the test set.

Measure accuracy on the results.
"""

import matplotlib.pyplot
import numpy as np

from evilsplit.adversarial import min_accuracy_split
from evilsplit.mnist import sample_images
from evilsplit.models import conv_classifier
from evilsplit.train import train_and_test

NUM_SAMPLES = 70000

def main():
    """
    Program entry-point.
    """
    print('Creating minimum-accuracy split...')
    train_indices, test_indices = min_accuracy_split()

    print('Training on adversarial split.')
    corrects, _ = train_and_test(train_indices, test_indices, architecture=conv_classifier)
    print('Mean accuracy: ' + str(np.mean(np.array(corrects).astype('float32'))))

    print('Plotting some samples.')
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    show_image_grid(sample_images(train_indices), figure=1)
    show_image_grid(sample_images(test_indices), figure=2)
    matplotlib.pyplot.show()

def show_image_grid(images, figure=1):
    """
    Show some images in a grid.
    """
    big_image = np.zeros((28*4, 28*4), dtype='float32')
    images = [i.reshape(28, 28) for i in images]
    for row in range(4):
        for col in range(4):
            big_image[row*28 : (row+1)*28, col*28 : (col+1)*28] = images[row*4 + col]
    matplotlib.pyplot.figure(figure)
    matplotlib.pyplot.imshow(big_image)

if __name__ == '__main__':
    main()
