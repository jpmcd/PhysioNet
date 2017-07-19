from __future__ import division

import input_data
import local_config
import layer_utils

import numpy as np
import tensorflow as tf

import os.path
import sys
import itertools


if __name__ == '__main__':

  path = local_config.STEVEN_PATH
  pickle_path = local_config.STEVEN_PICKLE_PATH
  n_folds = 4
  maxlen = 10000
  batch_size = 20
  num_epochs = 100
  conv1_channels = 32
  conv2_channels = 32
  out3_channels = 60
  l2_regularization = 0.002
  keep_probability = 0.9
  
  #Check if pickle file exists, else make file from .mat files
  print "Checking for data file"
  if not os.path.isfile(pickle_path):
    input_data.open_data(path, pickle_path)

  print "Splitting data into %d folds"
  split = input_data.split_datasets(pickle_path, n_folds)

  print "Concatenate %d folds for training, and save one for validation" 
  valid = [split[0][0], split[1][0]]
  train_x = itertools.chain([split[0][i] for i in len(split[0])])
  train_y = itertools.chain([split[1][i] for i in len(split[0])])
  train = [train_x, train_y]

  print "Cutting to length"
  train_x, train_y = input_data.prepare_dataset(train[0], train[1], maxlen)
  valid_x, valid_y = input_data.prepare_dataset(valid[0], valid[1], maxlen)
  n_train = len(train_y)
  n_valid = len(valid_y)

  #Perform signal normalization: max over multiple windows, take average
  win_length = 300 #number of samples in window
  stride = 300 #spaced apart
  if True:
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
      train_x[i] = x/avg_max

  #Build TensorFlow graph
  x = tf.placeholder(tf.float32, shape=[None, maxlen, 1], name='signal')
  y = tf.placeholder(tf.int32, shape=[None], name='labels')

  # Set up regularization.
  weight_decay = tf.constant(l2_regularization)
  l2_loss = 0
  
  #Inputs are given to 1D convolution
  with tf.variable_scope('conv1') as scope:
    [layer, loss] = layer_utils.add_convolutional_layers(
      input_tensor=x, conv_shape=[60, 1, conv1_channels], conv_stride=10, 
      window_shape=5, pool_strides=3, l2_regularization=weight_decay)
    l2_loss += loss

  with tf.variable_scope('conv2') as scope:
    [layer, loss] = layer_utils.add_convolutional_layers(
      input_tensor=layer, conv_shape=[5, conv1_channels, conv2_channels], conv_stride=5, 
      window_shape=3, pool_strides=2, l2_regularization=weight_decay)
    l2_loss += loss

  #Reshape layer to [batch, ndims]
  ndim = np.prod(layer.shape.as_list()[1:], dtype=np.int32)
  flat_layer = tf.reshape(layer, [-1, ndim])

  #Fully connected third layer
  with tf.variable_scope('full3') as scope:
    weights = tf.get_variable(name='weights', shape=[ndim, out3_channels],
      initializer=tf.truncated_normal_initializer(stddev=.001))
    bias = tf.get_variable(name='bias', shape=[out3_channels],
      initializer=tf.constant_initializer(0.1))
    full3 = tf.nn.relu(tf.matmul(flat_layer, weights)+bias, name='layer')
    keep_prob = tf.placeholder(tf.float32)
    out3 = tf.nn.dropout(full3, keep_prob)
    l2_loss += tf.scalar_mul(weight_decay, tf.nn.l2_loss(weights))

  #Softmax layer
  with tf.variable_scope('softmax') as scope:
    weights = tf.get_variable(name='weights', shape=[out3_channels, 4],
      initializer=tf.truncated_normal_initializer(stddev=1./out3_channels))
    bias = tf.get_variable(name='bias', shape=[4],
      initializer=tf.constant_initializer(0.0))
    softmax_logits = tf.add(tf.matmul(out3, weights), bias, name='softmax')
    l2_loss += tf.scalar_mul(weight_decay, tf.nn.l2_loss(weights))

  argmax = tf.argmax(softmax_logits, axis=1)

  #Cross entropy and loss with regularization term
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=softmax_logits, labels=y)
  avg_cross_entropy = tf.reduce_mean(cross_entropy)

  total_loss = tf.add(avg_cross_entropy, l2_loss, name='total_loss')

  #Training operation on loss
  tvars = tf.trainable_variables()
  grads = tf.gradients(total_loss, tvars)
  optimizer = tf.train.AdamOptimizer(1e-4)
  train_op = optimizer.apply_gradients(zip(grads, tvars))

  #Accuracy (max of logits vs. labels) used for both training and validation sets
  top_k = tf.nn.in_top_k(softmax_logits, y, k=1, name='top_k')
  #accuracy = tf.reduce_mean(top_k)
  accuracy = tf.reduce_mean(tf.cast(top_k, tf.float32))

  print 'accuracy', accuracy.shape.as_list()

  #Make session, initialize variables
  init = tf.global_variables_initializer()

  sess = tf.Session()  
  sess.run(init)

  #Execute graph, train on training data and test validation data
  print "Beginning training:"
  for epoch in range(num_epochs):
    print "Epoch %d" % epoch

    batch_inds = input_data.get_minibatch_inds(n_train, batch_size)
    valid_inds = input_data.get_minibatch_inds(n_valid, batch_size)

    n_batches = len(batch_inds)-1
    losses = np.zeros(n_batches)
    for i, inds in enumerate(batch_inds[:-1]):
      losses[i], _ = sess.run(
        [total_loss, train_op],
        feed_dict={x: train_x[inds], y: train_y[inds], keep_prob: keep_probability})
      print "batch %d / %d\r"%(i, n_batches) , 
      sys.stdout.flush()
    print ""

    avg_loss = np.mean(losses)

    acc, pred = sess.run(
      [accuracy, argmax],
      feed_dict={x: valid_x, y: valid_y, keep_prob: 1.0})

    cat_total = np.array([np.sum(valid_y==i) for i in range(4)])
    cat_corr = np.array([np.sum((valid_y==i)*(valid_y==pred)) for i in range(4)])
    Fscore = np.mean(cat_corr/cat_total)

    print ["%d / %d"%(cat_corr[i], cat_total[i]) for i in range(4)]
    print cat_corr/cat_total
    print "average loss: %f, validation accuracy: %f, F score: %f\n"%(avg_loss, acc, Fscore)

    #accuracies = np.zeros(len(valid_inds)-1)
    #for i, inds in enumerate(valid_inds[:-1]):
    #  acc, pred = sess.run([accuracy, argmax],
    #    feed_dict={x: valid_x[inds], y: valid_y[inds]})
    #  accuracies[i] = acc[0]

    #valid_acc = np.mean(accuracies)
    #print "average loss: %f, validation accuracy: %f\n"%(avg_loss, valid_acc)



