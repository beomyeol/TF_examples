#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

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
tf.app.flags.DEFINE_integer('max_steps', 100,
                            """Number of batches to run.""")

# To check the location of Saver
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Checkpoint options
tf.app.flags.DEFINE_integer('save_secs', None,
                            'Checkpoint save interval seconds.')

tf.app.flags.DEFINE_integer('save_steps', None,
                            'Checkpoint save interval steps')

# Summary
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

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

# Learning rate
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

# Optimization
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

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
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer

def get_train_op(dataset):
  global_step = tf.contrib.framework.get_or_create_global_step()

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

  network_fn = nets_factory.get_network_fn(
      FLAGS.model_name,
      num_classes=dataset.num_classes,
      weight_decay=FLAGS.weight_decay,
      is_training=True)

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

  logits, end_points = network_fn(images)

  if 'AuxLogits' in end_points:
    tf.losses.softmax_cross_entropy(
        end_points['AuxLogits'], labels,
        label_smoothing=FLAGS.label_smoothing, weights=0.4, scope='aux_loss')
  tf.losses.softmax_cross_entropy(
      logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)

  # Gather initial summaries
  summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

  # Add summaries for end_points
  for end_point in end_points:
    x = end_points[end_point]
    summaries.add(tf.summary.histogram('activations/' + end_point, x))
    summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                    tf.nn.zero_fraction(x)))

  # Add summaries for losses.
  for loss in tf.get_collection(tf.GraphKeys.LOSSES):
    summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

  # Add summaries for variables.
  for variable in slim.get_model_variables():
    summaries.add(tf.summary.histogram(variable.op.name, variable))

  # Configure the optimization
  learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
  optimizer = _configure_optimizer(learning_rate)

  losses = tf.get_collection(tf.GraphKeys.LOSSES)
  losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

  total_loss = tf.add_n(losses, name='total_loss')

  # Add total_loss to summary.
  summaries.add(tf.summary.scalar('total_loss', total_loss))

  # Compute gradients with respect to the loss.
  grads = optimizer.compute_gradients(total_loss)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  apply_gradients_op = optimizer.apply_gradients(grads, global_step=global_step)

  train_op = control_flow_ops.with_dependencies([apply_gradients_op], total_loss,
                                                name='train_op')

  summary_op = tf.summary.merge(list(summaries), name='summary_op')

  logging_tensors = {'step': global_step, 'loss': total_loss}

  return train_op, summary_op, logging_tensors

def train(dataset):
  train_op, summary_op, logging_tensors = get_train_op(dataset)

  hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps)]
  hooks.append(tf.train.LoggingTensorHook(tensors=logging_tensors,
                                          every_n_iter=10))

  hooks.append(tf.train.SummarySaverHook(save_steps=50,
                                         output_dir=FLAGS.train_dir,
                                         summary_op=summary_op))

  if FLAGS.save_steps or FLAGS.save_secs:
    hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.train_dir,
                                              save_steps=FLAGS.save_steps,
                                              save_secs=FLAGS.save_secs,
                                              saver=tf.train.Saver()))

  with tf.train.MonitoredTrainingSession(#checkpoint_dir=FLAGS.train_dir,
                                         save_summaries_steps=None, # disable default summary saver
                                         save_checkpoint_secs=None, # disable default checckpoint saver
                                         hooks=hooks) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(train_op)


def train_distributed_worker(cluster_spec, server, dataset):
  is_chief = (FLAGS.task_id == 0)

  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % FLAGS.task_id,
    cluster=cluster_spec)):

    train_op, summary_op, logging_tensors = get_train_op(dataset)

  hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps)]
  hooks.append(tf.train.LoggingTensorHook(tensors=logging_tensors,
                                          every_n_iter=10))

  if is_chief:
    hooks.append(tf.train.SummarySaverHook(save_steps=50,
                                           output_dir=FLAGS.train_dir,
                                           summary_op=summary_op))
    if FLAGS.save_steps or FLAGS.save_secs:
      hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.train_dir,
                                                save_steps=FLAGS.save_steps,
                                                save_secs=FLAGS.save_secs,
                                                saver=tf.train.Saver()))

  with tf.train.MonitoredTrainingSession(master=server.target,
                                         is_chief=is_chief,
                                         #checkpoint_dir=FLAGS.train_dir,
                                         save_summaries_steps=None, # disable default summary saver
                                         save_checkpoint_secs=None, # disable default checckpoint saver
                                         hooks=hooks) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(train_op)


def distributed_train(dataset):
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
    train_distributed_worker(cluster_spec, server, dataset)


def main(_):
  assert FLAGS.type in ['single', 'distributed'], 'type must be either "single" or "distributed"'

  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  dataset = dataset_factory.get_dataset(
      FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

  if FLAGS.save_steps and FLAGS.save_secs:
    raise ValueError('Either --save_steps or --save_secs must be set, not both')

  if FLAGS.type == 'single':
    train(dataset)
  else:
    distributed_train(dataset)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()