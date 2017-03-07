import argparse
import sys
import os

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_boolean(
    'ps_on_cpu', False, '')

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

FLAGS = None

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
    mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
    is_chief = (FLAGS.task_index == 0)

    ps_device = '/job:ps'
    if ps_on_cpu:
      ps_device += '/cpu'

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        ps_device=ps_device,
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      global_step = tf.contrib.framework.get_or_create_global_step()

      # Build model. CNN for MNIST
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
      y_ = tf.placeholder(tf.float32, [None, 10])

      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

    logging_tensors = {"loss": loss, "step": global_step}

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps)]

    scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(),
                                 init_fn=None,
                                 local_init_op=tf.group(tf.local_variables_initializer(),
                                                        tf.tables_initializer()),
                                 ready_op=tf.report_uninitialized_variables(),
                                 ready_for_local_init_op=None,
                                 summary_op=tf.summary.merge_all(),
                                 saver=tf.train.Saver())

    logdir = "/tmp/train_logs"

    # CheckpointSaverHook
    hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=logdir,
                                              save_secs=None,
                                              save_steps=100,
                                              scaffold=scaffold))

    # LoggingHook
    hooks.append(tf.train.LoggingTensorHook(logging_tensors,
                                            every_n_iter=1))

    # Summary writer
    hooks.append(tf.train.SummarySaverHook(save_steps=100,
                                           output_dir=logdir,
                                           summary_writer=None,
                                           scaffold=scaffold))

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=is_chief,
                                           scaffold=scaffold,
                                           save_checkpoint_secs=None, # disable default
                                           save_summaries_steps=None, # disable default
                                           checkpoint_dir=None,
                                           config=tf.ConfigProto(log_device_placement=True),
                                           hooks=hooks) as mon_sess:
      tf.logging.info('Start Training.')
      try:
        while not mon_sess.should_stop():
          # Run a training step asynchronously.
          # See `tf.train.SyncReplicasOptimizer` for additional details on how to
          # perform *synchronous* training.
          # mon_sess.run handles AbortedError in case of preempted PS.
          image, label = mnist.train.next_batch(FLAGS.batch_size)
          mon_sess.run(train_op, feed_dict={x: image, y_: label, keep_prob: 0.5})
      except:
        tf.logging.info('An exception is thrown.')
        raise

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  parser.add_argument(
      "--max_steps",
      type=int,
      default=1000,
      help="Maximum number of steps")
  parser.add_argument(
      "--batch_size",
      type=int,
      default=32,
      help="Batch size")
  FLAGS, unparsed = parser.parse_known_args()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)