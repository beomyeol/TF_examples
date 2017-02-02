#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("steps", 100, "Number of training steps")

FLAGS = tf.app.flags.FLAGS
batch_size = 100

def seconds_to_string(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  #initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main(_):
  hosts = FLAGS.hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"worker": hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name="worker",
                           task_index=FLAGS.task_index)

  is_chief = (FLAGS.task_index == 0)

  if not is_chief:
    server.join()
  else:
    # First worker
    with tf.device("/job:worker/task:%d" % 0):

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

    # Second worker
    with tf.device("/job:worker/task:%d" % 1):
      W_fc1 = weight_variable([7 * 7 * 64, 1024])
      b_fc1 = bias_variable([1024])

      h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

      keep_prob = tf.placeholder(tf.float32)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

      W_fc2 = weight_variable([1024, 10])
      b_fc2 = bias_variable([10])

      y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
      y_ = tf.placeholder(tf.float32, [None, 10])

      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
      #tf.summary.scalar("loss", loss)
      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.1).minimize(
          loss, global_step=global_step)

      #saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()

      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir="./train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             #saver=saver,
                             global_step=global_step)
                             #save_model_secs=600)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target, config=tf.ConfigProto(log_device_placement=True)) as sess:
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      print "Start session"
      start_time = time.clock()
      step = 0
      while not sv.should_stop() and step < FLAGS.steps:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        batch = mnist.train.next_batch(batch_size)
        if step % 50 == 0:
          train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
          print("step %d, training accuracy %g" % (step, train_accuracy))
        _, step = sess.run([train_op, global_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
          #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

      end_time = time.clock()
      #print "Training complete"
      print "Trianing time: ", seconds_to_string(end_time - start_time)
      start_time = time.clock()

      train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
      print("test accuracy %g" % (train_accuracy))
      print "Testing time: ", seconds_to_string(time.clock() - start_time)

    print "Processing complete"
    # Ask for all the services to stop.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()
