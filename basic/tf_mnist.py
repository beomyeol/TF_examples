#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

log_period = 1
max_step = 300
batch_size = 100

if __name__ == "__main__":
  np.set_printoptions(threshold=np.nan)

  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  # with tf.device('/cpu:0'):
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
  y_ = tf.placeholder(tf.float32, [None, 10])

  # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
  # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
  sigmoid = tf.sigmoid(y)
  loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=sigmoid, weights=0.5))

  train_step = tf.train.GradientDescentOptimizer(1.5).minimize(loss)
  init = tf.global_variables_initializer()

  sess_config = None
  sess_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})

  sess = tf.InteractiveSession(config=sess_config)
  sess.run(init)

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  for i in range(max_step):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    # print "batch_xs=%s, batch_ys=%s" % (batch_xs, batch_ys)
    _, sigmoid_out, y_out, new_W, current_loss = sess.run([train_step, sigmoid, y, W, loss], feed_dict={x: batch_xs, y_: batch_ys})
    # print "y=%s" % y_out
    # print "sigmoid=%s" % sigmoid_out
    # print "new_W=%s" % new_W
    # print "loss=%f" % current_loss
    if i % log_period == 0:
      print 'step=%d, accuracy=%f' % (i,
        sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

  sess.close()
