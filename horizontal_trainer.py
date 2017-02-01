#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("steps", 100, "Number of training steps")

FLAGS = tf.app.flags.FLAGS
batch_size = 100

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main(_):
  hosts = FLAGS.hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"worker": hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name="worker",
                           task_index=FLAGS.task_index)

  is_chief = (FLAGS.task_index == 1)

  if not is_chief:
    server.join()
  else:
    # First worker
    with tf.device("/job:worker/task:%d" % 0):
      x = tf.placeholder(tf.float32, [None, 784])
      W1 = weight_variable([784, 100])
      b1 = bias_variable([100])
      y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    # Second worker
    with tf.device("/job:worker/task:%d" % 1):
      W2 = weight_variable([100, 10])
      b2 = bias_variable([10])
      y = tf.matmul(y1, W2) + b2
      y_ = tf.placeholder(tf.float32, [None, 10])

      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.1).minimize(
          loss, global_step=global_step)

      saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()

      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 1),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             #saver=saver,
                             global_step=global_step)
                             #save_model_secs=600)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      print "Start session"
      step = 0
      while not sv.should_stop() and step < FLAGS.steps:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        batch = mnist.train.next_batch(batch_size)
        _, step = sess.run([train_op, global_step], feed_dict={x: batch[0], y_: batch[1]})

      print "Training complete"
      print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    print "Processing complete"
    # Ask for all the services to stop.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()
