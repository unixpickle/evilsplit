"""
MNIST-specific APIs.
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100

def fully_connected_classifier(input_ph):
    """
    Apply a simple fully-connected network to the image to
    produce output logits.
    """
    hidden = tf.contrib.layers.fully_connected(input_ph, 300)
    return tf.contrib.layers.fully_connected(hidden, 10, activation_fn=None)

def train_and_test(train_indices, test_indices, architecture=fully_connected_classifier):
    """
    Train an MNIST classifier on the training data and
    compute its accuracy on the test data.

    Args:
      train_indices: indices of training samples (0, 70K].
      test_indices: indices of testing samples (0, 70K].
      architecture: the input->logit function to train.

    Returns:
      A tuple (corrects, loss), where:
        corrects: a list of booleans indicating if each
          test sample was correctly classified.
        losses: a list of losses, one per test sample.
    """
    # pylint: disable=E1129
    with tf.Graph().as_default():
        with tf.Session() as sess:
            input_ph = tf.placeholder(tf.float32, shape=(None, 784))
            logits = architecture(input_ph)
            _train(sess, input_ph, logits, *_get_data(train_indices))
            return _test(sess, input_ph, logits, *_get_data(test_indices))

def sample_images(indices):
    """
    Get the images for the sample indices.
    """
    return _get_data(indices)[0]

def _train(sess, input_ph, logits, training_ins, training_outs):
    """
    Train the classifier.
    """
    labels_ph = tf.placeholder(tf.float32, shape=(None, 10))
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_ph, logits=logits)
    optim = tf.train.AdamOptimizer().minimize(tf.reduce_mean(loss))
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        shuffled = np.random.permutation(len(training_ins))
        shuf_ins, shuf_outs = training_ins[shuffled], training_outs[shuffled]
        for i in range(0, len(training_ins), BATCH_SIZE):
            ins, outs = shuf_ins[i : i+BATCH_SIZE], shuf_outs[i : i+BATCH_SIZE]
            sess.run(optim, feed_dict={input_ph: ins, labels_ph: outs})

# pylint: disable=R0914
def _test(sess, input_ph, logits, testing_ins, testing_outs):
    """
    Compute correct booleans and loss values for the
    testing samples.
    """
    assert len(testing_ins) % BATCH_SIZE == 0
    one_hot_out = tf.one_hot(tf.argmax(logits, axis=-1), 10)
    labels_ph = tf.placeholder(tf.float32, shape=(None, 10))
    corrects = tf.greater(tf.reduce_sum(one_hot_out * labels_ph, axis=-1), 0)
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels_ph, logits=logits)
    all_corrects = []
    all_losses = []
    for i in range(0, len(testing_ins), BATCH_SIZE):
        ins, outs = testing_ins[i : i+BATCH_SIZE], testing_outs[i : i+BATCH_SIZE]
        corr_out, loss_out = sess.run((corrects, losses),
                                      feed_dict={input_ph: ins, labels_ph: outs})
        all_corrects.extend(corr_out)
        all_losses.extend(loss_out)
    return all_corrects, all_losses

_MNIST_IMAGES = None
_MNIST_LABELS = None

def _get_data(indices):
    """
    Get (inputs, outputs) for the training samples.
    """
    # pylint: disable=W0603
    global _MNIST_IMAGES
    global _MNIST_LABELS
    if _MNIST_LABELS is None:
        data = input_data.read_data_sets('MNIST_data', one_hot=True)
        _MNIST_LABELS = np.array(list(data.train.labels) + list(data.test.labels))
        _MNIST_IMAGES = np.array(list(data.train.images) + list(data.test.images))
    return _MNIST_IMAGES[indices], _MNIST_LABELS[indices]
