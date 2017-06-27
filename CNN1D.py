from __future__ import division

import numpy as np
import tensorflow as tf

import os.path
import input_data
import sys


if __name__ == '__main__':

  path = '/home/mcdonald/Desktop/training2017/'
  pickle_path = '/scratch/PhysioNet/data.pkl'
  valid_pct = 1./4
  maxlen = 10000
  batch_size = 20
  num_epochs = 100
  conv1_channels = 64
  conv2_channels = 64
  out3_channels = 60
  
  #Check if pickle file exists, else make file from .mat files
  print "Checking for data file"
  if not os.path.isfile(pickle_path):
    input_data.open_data(path, pickle_path)

  print "Splitting training and validation sets"
  train, valid = input_data.split_datasets(pickle_path, valid_pct)

  print "Cutting to length"
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
    kernel = tf.get_variable(name='weights', shape=[60,1,conv1_channels],
      initializer=tf.truncated_normal_initializer(stddev=.001))
    bias = tf.get_variable(name='bias', shape=[conv1_channels],
      initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv1d(x, kernel, stride=10, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
    layer = tf.nn.pool(relu, window_shape=[5], pooling_type='MAX',
      padding='SAME', strides=[3], name='layer')

  with tf.variable_scope('conv2') as scope:
    kernel = tf.get_variable(name='weights',
      shape=[5,conv1_channels,conv2_channels], 
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
    full3 = tf.nn.relu(tf.matmul(flat_layer, weights)+bias, name='layer')

  #Softmax layer
  with tf.variable_scope('softmax') as scope:
    weights = tf.get_variable(name='weights', shape=[out3_channels, 4],
      initializer=tf.truncated_normal_initializer(stddev=1./out3_channels))
    bias = tf.get_variable(name='bias', shape=[4],
      initializer=tf.constant_initializer(0.0))
    softmax_logits = tf.add(tf.matmul(full3, weights), bias, name='softmax')

  argmax = tf.argmax(softmax_logits, axis=1)

  #Cross entropy and loss with regularization term
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=softmax_logits, labels=y)
  avg_cross_entropy = tf.reduce_mean(cross_entropy)

  weight_decay = tf.constant(0.01)
  l2_loss = tf.nn.l2_loss(weights)
  total_loss = tf.add(avg_cross_entropy, tf.scalar_mul(weight_decay, l2_loss),
    name='total_loss')

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
      losses[i], _ = sess.run([total_loss, train_op],
        feed_dict={x: train_x[inds], y: train_y[inds]})
      print "batch %d / %d\r"%(i, n_batches) , 
      sys.stdout.flush()
    print ""

    avg_loss = np.mean(losses)

    acc, pred = sess.run([accuracy, argmax],
      feed_dict={x: valid_x, y: valid_y})

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



