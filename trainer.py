#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
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
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    is_chief = (FLAGS.task_index == 0)

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      x = tf.placeholder(tf.float32, [None, 784])
      y_ = tf.placeholder(tf.float32, [None, 10])

      # Small model
      # W = tf.Variable(tf.zeros([784, 10]))
      # b = tf.Variable(tf.zeros([10]))
      # y = tf.matmul(x, W) + b
      # y_ = tf.placeholder(tf.float32, [None, 10])

      # Large model
      W1 = weight_variable([784, 100])
      b1 = bias_variable([100])
      y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
      W2 = weight_variable([100, 10])
      b2 = bias_variable([10])
      y = tf.matmul(y1, W2) + b2

      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.1).minimize(
          loss, global_step=global_step)

      saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()

      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir="./train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      print "Start loop"
      step = 0
      while not sv.should_stop() and step < FLAGS.steps:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        batch = mnist.train.next_batch(batch_size)
        _, step = sess.run([train_op, global_step], feed_dict={x: batch[0], y_: batch[1]})

      print "Training Complete. Check accuracy"
      print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    # Ask for all the services to stop.
    print "Training Complete. Ask for all the service to stop"
    sv.stop()

if __name__ == "__main__":
  tf.app.run()
