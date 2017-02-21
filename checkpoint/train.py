#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from nets import nets_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('type', 'single', 'One of "single" and "distributed"')
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_integer('task_id', 0,
                            'Task ID of the worker/replica running the training.')

tf.app.flags.DEFINE_string('train_dir', '/tmp/train_logs',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")

# To check the location of Saver
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Checkpoint options
tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60,
                            'Save interval seconds.')

# Dataset
tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

# Optimization
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')


FLAGS = tf.app.flags.FLAGS

def _configure_learning_rate(num_samples_per_epoch, global_step):
  # TODO: can be extended
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
  return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')

def _configure_optimizer(learning_rate):
  # TODO: can be extended
  optimizer = tf.train.RMSPropOptimizer(
      learning_rate,
      decay=FLAGS.rmsprop_decay,
      momentum=FLAGS.rmsprop_momentum,
      epsilon=FLAGS.opt_epsilon)
  return optimizer

def clone_fn(batch_queue):


def train_distributed_worker(cluster_spec, dataset):
  is_cheif = (FLAGS.task_id == 0)

  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % FLAGS.task_id,
    cluster=cluster_spec)):

    global_step=tf.contrib.framework.get_or_create_global_step()

    preprocessing_name = FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)

    # Data provider
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=10 * FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])

    train_image_size = network_fn.default_image_size

    image = image_preprocessing_fn(image, train_image_size, train_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
    labels = slim.one_hot_encoding(labels, dataset.num_classes)
    # batch_queue = slim.prefetch_queue.prefetch_queue(
    #     [images, labels], capacity=2 * deploy_config.num_clones)

    # Build model
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=dataset.num_classes,
        weight_decay=FLAGS.weight_decay,
        is_training=True)

    logits, end_points = network_fn(images)

    if 'AuxLogits' in end_points:
        slim.losses.softmax_cross_entropy(
            end_points['AuxLogits'], labels,
            label_smoothing=FLAGS.label_smoothing, weight=0.4, scope='aux_loss')
      slim.losses.softmax_cross_entropy(
          logits, labels, label_smoothing=FLAGS.label_smoothing, weight=1.0)

    # Gather initial summaries
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for end_points
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.histogram_summary('activations/' + end_point, x))
      summaries.add(tf.scalar_summary('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.scalar_summary('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.histogram_summary(variable.op.name, variable))

    # Configure the optimization
    learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
    optimizer = _configure_optimizer(learning_rate)

    losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
    losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    total_loss = tf.add_n(losses, name='total_loss')

    # Add total_loss to summary.
    summaries.add(tf.scalar_summary('total_loss', total_loss,
                                    name='total_loss'))

    # Compute gradients with respect to the loss.
    grads = opt.compute_gradients(total_loss)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.histogram_summary(var.op.name + '/gradients', grad)

    apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

    train_op = control_flow_ops.with_dependencies([apply_gradients_op], total_loss,
                                                  name='train_op')

  hooks=[tf.train.StopAtStepHook(last_step=100)]

  with tf.train.MonitoredTrainingSetssion(master=server.target,
                                          is_chief=is_chief,
                                          checkpoint_dir=FLAGS.train_dir,
                                          hooks=hooks) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(train_op)


def distributed_train():
  assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

  ps_hosts = FLAGS.ps_hosts.split(',')
  worker_hosts = FLAGS.worker_hosts.split(',')
  tf.logging.info('PS hosts are: %s' % ps_hosts)
  tf.logging.info('Worker hosts are: %s' % worker_hosts)

  cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
                                       'worker': worker_hosts})

  server = tf.train.Server(cluster_spec,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_id)

  if FLAGS.job_name == 'ps':
    server.join()
  else:
    if not FLAGS.dataset_dir:
      raise ValueError('You must supply the dataset directory with --dataset_dir')

    dataset = dataset_factory.get_dataset(
      FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    train_distributed_worker(cluster_spec, dataset)


def main(_):
  assert FLAGS.type in ['single', 'distributed'], 'type must be either "single" or "distributed"'

  if FLAGS.type == 'single':
    single_train()
  else:
    distributed_train()

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()