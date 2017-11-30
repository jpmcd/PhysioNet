# Convenience functions for putting together TensorFlow layers.

import numpy as np
import tensorflow as tf


def signal_normalization(train_x, win_length, stride):
  n_train = len(train_x)
  normalized_x = np.empty(train_x.shape, dtype=float)

  for i in range(n_train):
    x = train_x[i]
    sig_length = len(x)
    n_slices = int(np.floor((sig_length - (win_length - stride)) / stride))
    maxes = np.zeros(n_slices)

    for j in range(n_slices):
      start = j*stride
      window = x[start:start+win_length]
      maxes[j] = np.max(window)

    avg_max = np.mean(maxes)
    normalized_x[i] = x/avg_max

  return normalized_x


def add_convolutional_layers(input_tensor, conv_shape, conv_stride, window_shape, pool_strides,
                            l2_regularization):
    # Adds a convolutional layer to the network. Returns the L2 norm of the convolutional kernel.
    kernel = tf.get_variable(name='weights', shape=conv_shape,
      initializer=tf.truncated_normal_initializer(stddev=.001))
    bias = tf.get_variable(name='bias', shape=[conv_shape[-1]],
      initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv1d(input_tensor, kernel, stride=conv_stride, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
    layer = tf.nn.pool(relu, window_shape=[window_shape], pooling_type='MAX',
      padding='SAME', strides=[pool_strides], name='layer')
    l2_loss = tf.scalar_mul(l2_regularization, tf.nn.l2_loss(kernel))
    return [layer, l2_loss]
