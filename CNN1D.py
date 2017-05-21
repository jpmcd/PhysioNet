from __future__import division

import numpy as np
import tensorflow as tf

import os.path
import input_data


if __name__ == '__main__':

  path = '/home/mcdonald/Desktop/training2017/'
  pickle_path = '/scratch/PhysioNet/data.pkl'
  valid_pct = 1./4
  maxlen = 10000
  batch_size = 20
  num_epochs = 100
  conv1_channels = 32
  conv2_channels = 32
  out3_channels = 60
  
  #Check if pickle file exists, else make file from .mat files
  if not os.path.isfile(pickle_path):
    input_data.open_data(path, pickle_path)

  train, valid = input_data.split_datasets(pickle_path, valid_pct)

  train_x, train_y = input_data.prepare_dataset(train[0], train[1], maxlen)
  valid_x, valid_y = input_data.prepare_dataset(valid[0], valid[1], maxlen)
  n_train = len(train_y)
  n_valid = len(valid_y)

  #Perform signal normalization? Are signals normalized?

  #Build TensorFlow graph
  x = tf.placeholder(tf.float32, shape=[None, maxlen, 1], name='signal')
  y = tf.placeholder(tf.int32, shape=[None], name='labels')

  
  #Inputs are given to 1D convolution
  with tf.variable_scope('conv1') as scope:
    kernel = tf.get_variable(name='weights', shape=[30,1,conv1_channels],
      initializer=tf.truncated_normal_initializer(stddev=.001))
    bias = tf.get_variable(name='bias', shape=[conv1_channels],
      initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv1d(x, kernel, stride=10, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
    layer = tf.nn.pool(relu, window_shape=[5], pooling_type='MAX',
      padding='SAME', strides=[3], name='layer')

  with tf.variable_scope('conv2') as scope:
    kernel = tf.get_variable(name='weights',
      shape=[10,conv1_channels,conv2_channels], 
      initializer=tf.truncated_normal_initializer(stddev=.001))
    bias = tf.get_variable(name='bias', shape=[conv2_channels],
      initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv1d(layer, kernel, stride=5, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
    layer = tf.nn.pool(relu, window_shape=[3], pooling_type='MAX',
      padding='SAME', strides=[2], name='layer')

  #Reshape layer to [batch, ndims]
  ndim = np.prod(layer.shape.as_list()[1:], dtype=np.int32)
  flat_layer = tf.reshape(layer, [-1, ndim])

  #Fully connected third layer
  with tf.variable_scope('full3') as scope:
    weights = tf.get_variable(name='weights', shape=[ndim, out3_channels],
      initializer=tf.truncated_normal_initializer(stddev=.001))
    bias = tf.get_variable(name='bias', shape=[out3_channels],
      initializer=tf.constant_initializer(0.1))
    full3 = tf.relu(tf.matmul(flat_layer, weights)+bias, name='layer')

  #Softmax layer
  with tf.variable_scope('softmax') as scope:
    weights = tf.get_variable(name='weights', shape=[out3_channels, 4],
      initializer=tf.truncated_normal_initializer(stddev=1./out3_channels))
    bias = tf.get_variable(name='bias', shape=4,
      initializer=tf.constant_initializer(0.0))
    softmax_logits = tf.add(tf.matmul(full3, weights), bias, name='softmax')

  #Cross entropy and loss with regularization term
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=softmax_logits, labels=labels)
  avg_cross_entropy = tf.reduce_mean(cross_entropy)

  weight_decay = tf.constant(0.01)
  total_loss = tf.add(avg_cross_entropy, tf.scalar_mul(weight_decay, weights),
    name='total_loss')

  #Training operation on loss
  tvars = tf.trainable_variables()
  grads = tf.gradients(total_loss, tvars)
  optimizer = tf.train.AdamOptimizer(1e-4)
  train_op = optimizer.apply_gradients(zip(grads, tvars))

  #Accuracy (max of logits vs. labels) used for both training and validation sets


  #Execute graph, train on training data and test validation data
  batch_inds = input_data.get_minibatch_inds(n_train, batch_size)

  for inds in batch_inds:
    sess.run([loss, train, correct], feed_dict={x: train_x[inds], y: train_y[inds]})




  

