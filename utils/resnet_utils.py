"""Utilities for building v2-variant of ResNets.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers


class Block(collections.namedtuple("Block", ["scope", "unit_fn", "args"])):
  """Contains definitions of a ResNet Block, where

  `scope`: The scope of the block.
  `unit_fn`: The ResNet unit function.
  `args`: The arguments of `unit_fn`. A list with length equal to the number
    of units in a block. Contains a dict mapping from arg name to arg value.
  """


@add_arg_scope
def stack_blocks(net, blocks):
  """Stacks ResNet blocks."""
  for block in blocks:
    with tf.variable_scope(block.scope, "block", [net]):
      for i, unit_args in enumerate(block.args):
        with tf.variable_scope("unit_%d" % (i + 1), values=[net]):
          net = block.unit_fn(net, **unit_args)

  return net


def resnet_arg_scope(weight_decay=2e-4,
                     batch_norm_decay=0.99,
                     batch_norm_epsilon=1e-3,
                     batch_norm_scale=True,
                     batch_norm_fused=True,
                     random_seed=None):
  """Defines the default ResNet arguments."""
  batch_norm_params = {
      "decay": batch_norm_decay,
      "epsilon": batch_norm_epsilon,
      "scale": batch_norm_scale,
      "fused": batch_norm_fused,
      "updates_collections": tf.GraphKeys.UPDATE_OPS}

  with arg_scope([layers_lib.conv2d],
      weights_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
      weights_initializer=tf.contrib.layers.xavier_initializer(seed=random_seed),
      activation_fn=tf.nn.relu,
      normalizer_fn=layers.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc
