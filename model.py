"""The preactivation variant of Residual networks for classifying 32 by 32
images of the CIFAR10 dataset.

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils

import data
from utils import resnet_utils

resnet_arg_scope = resnet_utils.resnet_arg_scope


class ResNetModel(object):
  """ResNet Model builder."""
  def __init__(self, hparams, dataset, mode):
    self._mode = mode
    self._is_training = self._mode == tf.contrib.learn.ModeKeys.TRAIN
    self._shortcut_conn = hparams.shortcut_conn

    with arg_scope(resnet_arg_scope(
        weight_decay=hparams.weight_decay,
        batch_norm_epsilon=hparams.epsilon,
        batch_norm_fused=hparams.fused,
        batch_norm_is_training=self._is_training,
        random_seed=hparams.random_seed)):
      self._logits = self._build_graph(hparams, dataset)

  def _build_graph(self, hparams, dataset, reuse=None):
    """Builds the graph for the forward pass (from `dataset.inputs` to `logits`)
    """ 
    inputs = dataset.inputs
    num_layers = hparams.num_layers
    shortcut_conn = self._shortcut_conn

    if num_layers not in (20, 32, 44, 56, 110):
      raise ValueError("`num_layers` must be 20, 32, 44, 56 or 110.")

    num_units = (num_layers - 2) // 6
    scope = "resnet_v2_cifar10_%d" % num_layers

    blocks = [resnet_v2_block("block1", num_units, 16, 1, shortcut_conn, True),
        resnet_v2_block("block2", num_units, 32, 2, shortcut_conn, False),
        resnet_v2_block("block3", num_units, 64, 2, shortcut_conn, False)]

    with tf.variable_scope(scope, "resnet_v2", [inputs], reuse=reuse):
      net = inputs

      net = layers_lib.conv2d(net, 16, 3, 1, "SAME", 
          activation_fn=None,
          normalizer_fn=None,
          biases_initializer=None,
          scope="init_conv")

      net = resnet_utils.stack_blocks(net, blocks)

      net = layers.batch_norm(
          net, activation_fn=tf.nn.relu, scope="postnorm")
      net = tf.reduce_mean(
          net, [1, 2], name="global_average_pooling", keepdims=True)
      net = layers_lib.conv2d(net, 10, 1, 1, 
          activation_fn=None,
          normalizer_fn=None,
          weights_initializer=tf.initializers.variance_scaling(
              distribution="uniform",
              seed=hparams.random_seed),
          scope="logits")
      logits = tf.squeeze(net, axis=[1, 2])

    return logits


@add_arg_scope
def unit_fn_v2(inputs,
               depth,
               stride,
               shortcut_conn,
               shortcut_from_preact,
               scope=None):
  """The v2 variant of ResNet unit with BN-relu preceeding convolution.
  See ref [2].
  """
  with tf.variable_scope(scope, "unit_fn_v2", [inputs]):
    depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = layers.batch_norm(
        inputs, activation_fn=tf.nn.relu, scope="preact")

    shortcut = preact if shortcut_from_preact else inputs

    if depth != depth_in:
      with tf.name_scope("shortcut"):
        shortcut = layers.avg_pool2d(shortcut, [2, 2], stride=2)
        shortcut = tf.pad(
            shortcut, [[0, 0], [0, 0], [0, 0], [(depth - depth_in) // 2] * 2])

    residual = layers_lib.conv2d(preact, depth, 3, stride, scope="conv1")
    residual = layers_lib.conv2d(residual, depth, 3, 1,
        normalizer_fn=None, activation_fn=None, biases_initializer=None,
        scope="conv2")

    output = residual + shortcut if shortcut_conn else residual

    return output 


def resnet_v2_block(
    scope, num_units, depth, stride, shortcut_conn, shortcut_from_preact):
  """Helper for creating ResNet v2 block definition."""
  args = [{
      "depth": depth,
      "stride": stride if i == 0 else 1,
      "shortcut_conn": shortcut_conn,
      "shortcut_from_preact": shortcut_from_preact if i == 0 else False}
      for i in range(num_units)]
  return resnet_utils.Block(scope, unit_fn_v2, args)
