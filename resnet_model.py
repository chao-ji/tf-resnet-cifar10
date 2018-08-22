from abc import ABCMeta
from abc import abstractproperty

import tensorflow as tf

from utils import resnet_utils
from utils import model_utils

slim = tf.contrib.slim


class ResnetPredictionModel(object):
  """Resnet model for classifying 32x32 images in Cifar10 dataset.

  Implements a `predict` method that runs the input images through the 
  forward pass to get the prediction logits to be consumed by subclasses,
  Trainer and Evaluator. They need to implement `train` and 'evaluator' method,
  respectively.
  """

  __metaclass__ = ABCMeta

  def __init__(self,
               conv_hyperparams_fn,
               num_layers,
               shortcut_connection=True,
               reuse_weights=None):
    """Constructor.

    Args:
      conv_hyperparams_fn: a callable, that takes no arguments and generates
        a dict to be passed to slim.arg_scope.
      num_layers: int scalar, the num of weighted layers.
      shortcut_connection: bool scalar, whether to add shortcut connection in
        each Resnet unit. If False, degenerates to a 'Plain network'.
      reuse_weights: None or bool scalar, whether to reuse weights.
    """
    if num_layers not in (20, 32, 44, 56, 110):
      raise ValueError('num_layers must be one of 20, 32, 44, 56 or 110.')
    # num of units per block
    num_units = (num_layers - 2) // 6

    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._num_layers = num_layers
    self._shortcut_connection = shortcut_connection
    self._reuse_weights = reuse_weights
    self._scope = 'Resnet_v2_Cifar10_{}'.format(num_layers)
    self._blocks = [resnet_utils.resnet_v2_block(
                        scope='block1', 
                        num_units=num_units, 
                        depth=16,
                        stride=1,
                        shortcut_connection=shortcut_connection,
                        shortcut_from_preact=True),
                    resnet_utils.resnet_v2_block(
                        scope='block2',
                        num_units=num_units,
                        depth=32,
                        stride=2,
                        shortcut_connection=shortcut_connection,
                        shortcut_from_preact=False),
                    resnet_utils.resnet_v2_block(
                        scope='block3',
                        num_units=num_units,
                        depth=64,
                        stride=2,
                        shortcut_connection=shortcut_connection,
                        shortcut_from_preact=False)]
 
  @abstractproperty
  def is_training(self):
    """Returns a bool scalar indicating if model is in training mode.
    """
    pass

  @abstractproperty
  def mode(self):
    """Returns a string scalar indicating the mode of model (train or eval).
    """
    pass  
  
  def predict(self, images):
    """Generates prediction logits from input images.

    Args:
      images: a float tensor with shape [batch, height, width, channels]

    Returns:
      logits: a float tensor with shape [batch, 10].
    """
    with slim.arg_scope([slim.batch_norm], is_training=self.is_training):
      with slim.arg_scope(self._conv_hyperparams_fn()):
        with tf.variable_scope(self._scope,
                               values=[images],
                               reuse=self._reuse_weights):
          net = images
          # initial conv has no activation, no batch norm and no bias:
          net = slim.conv2d(net, 16, 3, 1, 'SAME',
              activation_fn=None,
              normalizer_fn=None,
              biases_initializer=None,
              scope='init_conv')

          net = resnet_utils.stack_blocks(net, self._blocks)

          net = slim.batch_norm(
              net, activation_fn=tf.nn.relu, scope='postnorm')
          net = tf.reduce_mean(
              net, [1, 2], name='global_average_pooling', keepdims=True)
          # final conv simply performs an affine transformation:
          net = slim.conv2d(net, 10, 1, 1,
              activation_fn=None,
              normalizer_fn=None,
              scope='logits')
          logits = tf.squeeze(net, axis=[1, 2])
          return logits

  def check_dataset_mode(self, dataset):
    """Checks if mode (train, eval, or infer) of dataset and model match."""
    if dataset.mode != self.mode:
      raise ValueError('mode of dataset({}) and model({}) do not match.'
          .format(dataset.mode, self.mode))


class ResnetModelTrainer(ResnetPredictionModel):
  """Performs training."""
  @property
  def is_training(self):
    return True

  @property
  def mode(self):
    return tf.contrib.learn.ModeKeys.TRAIN

  def train(self, path, dataset, optimizer, learning_rate):
    """Adds train related ops to the graph.

    Args:
      path: a string scalar, the path to the directory containing cifar10
        binary files. 
      dataset: a dataset.Cifar10Dataset instance, the input data generate.
      optimizer: an optimizer instance, that computes and applies gradient
        updates.
      learning_rate: a float scalar tensor, the learning rate.

    Returns:
      grouped_update_op: a grouped op, that includes batch norm update ops
        and gradient update ops.
      total_loss: a float scalar tensor, the total loss.
      accuracy: a float scalar tensor, classification accuracy on each
        mini batch training set.
      summary: a scalar tensor containing the protobuf message of summary.
      global_step: an int scalar, global step.
    """
    self.check_dataset_mode(dataset)

    tensor_dict = dataset.get_tensor_dict(path)
    logits = self.predict(tensor_dict['images'])
    total_loss = model_utils.compute_loss(tensor_dict['labels'], logits)
    accuracy = model_utils.compute_accuracy(tensor_dict['labels'], logits)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.train.get_or_create_global_step()
    grads_and_vars = optimizer.compute_gradients(total_loss)
    grad_update_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)
    update_ops.append(grad_update_op)

    grouped_update_op = tf.group(*update_ops, name='update_barrier') 

    summary = model_utils.build_training_summary(total_loss,
                                                 accuracy,
                                                 learning_rate)

    return grouped_update_op, total_loss, accuracy, summary, global_step


class ResnetModelEvaluator(ResnetPredictionModel):
  """Performs evaluation."""
  @property
  def is_training(self):
    return False

  @property
  def mode(self):
    return tf.contrib.learn.ModeKeys.EVAL

  def evaluate(self, path, dataset):
    """Adds evaluation related ops to the graph.

    Args:
      path: a string scalar, the path to the directory containing cifar10
        binary files. 
      dataset: a dataset.Cifar10Dataset instance, the input data generate.

    Returns:
      total_loss: a float scalar tensor, the total loss.
      accuracy: a float scalar tensor, classification accuracy on each
        mini batch evaluation set (depends on `batch_size` of the dataset).
    """
    self.check_dataset_mode(dataset)

    tensor_dict = dataset.get_tensor_dict(path)
    logits = self.predict(tensor_dict['images'])
    total_loss = model_utils.compute_loss(tensor_dict['labels'], logits)
    accuracy = model_utils.compute_accuracy(tensor_dict['labels'], logits)

    return total_loss, accuracy 

