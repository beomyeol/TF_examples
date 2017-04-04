#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  # with tf.device('/cpu:0'):
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
  y_ = tf.placeholder(tf.float32, [None, 10])

  # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
  cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))

  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  init = tf.global_variables_initializer()

  sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
  sess.run(init)

  log_period = 1

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % log_period == 0:
      print 'step=%d, accuracy=%f' % (i,
        sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

  sess.close()
