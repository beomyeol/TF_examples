#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

tf.app.flags.DEFINE_integer("steps", 100, "Number of training steps")

FLAGS = tf.app.flags.FLAGS
batch_size = 100

def seconds_to_string(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

def weight_variable(shape):
  #initial = tf.truncated_normal(shape, stddev=0.1)
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main(_):
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  x = tf.placeholder(tf.float32, [None, 784])

  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])

  x_image = tf.reshape(x, [-1,28,28,1])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  #y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
  #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    start_time = time.clock()
    sess.run(tf.initialize_all_variables())
    end_time = time.clock()
    print "Initialization time: ", seconds_to_string(end_time - start_time)
    start_time = time.clock()

    for i in range(FLAGS.steps):
      batch = mnist.train.next_batch(batch_size)
      sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      if i % 50 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
      #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    end_time = time.clock()
    print "Trianing time: ", seconds_to_string(end_time - start_time)
    start_time = time.clock()

    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    print "Testing time: ", seconds_to_string(time.clock() - start_time)

if __name__ == "__main__":
  tf.app.run()