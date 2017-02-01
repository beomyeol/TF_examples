#!/usr/bin/env python

import tensorflow as tf

if __name__ == "__main__":
  with tf.device('/cpu:0'):
    x = tf.constant([[1., 2.]])
    W = tf.constant([[3.], [4.]])

  with tf.device('/gpu:0'):
    y = tf.matmul(x, W)

  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    # tf.global_variables_initializer().run()
    result = session.run(y)
    print result
