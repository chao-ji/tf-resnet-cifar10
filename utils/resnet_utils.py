import collections

import tensorflow as tf

slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """Wrapper of a namedtuple containing the following properties of of a 
    Resnet Block. 

    scope: string scalar, scope name of block.
    unit_fn: a callable, the Resnet unit function.
    args: a list of dict mapping from arg name to arg value, contaning the 
      kwargs for each `unit_fn` in this block.
  """
  pass


def stack_blocks(net, blocks):
  """Stacks up multiple Resnet blocks to form the backbone of Resnet.

  Args:
    net: a float tensor with shape [batch, height, width, channels],
      the input feature map.
    blocks: a list of `Block` instances containing the block specs.

  Returns:
    net: a float tensor with shape [batch, height, width, channels],
      the output feature map.
  """
  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]):
      for i, unit_args in enumerate(block.args):
        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          net = block.unit_fn(net, **unit_args)

  return net


def build_arg_scope_fn(weight_decay=2e-4,
                       batch_norm_decay=0.99,
                       batch_norm_epsilon=1e-3,
                       batch_norm_center=True,
                       batch_norm_scale=True,
                       batch_norm_fused=True):
  """Builds a function that generates a callable that produces input to
  slim.arg_scope for specifying argument scope of slim.conv2d op.

  Args:
    weight_decay: float scalar, weight for l2 regularization.
    batch_norm_decay: float scalar, the moving avearge decay.
    batch_norm_epsilon: float scalar, small value to avoid divide by zero.
    batch_norm_center: bool scalar, whether to center in the batch norm.
    batch_norm_scale: bool scalar, whether to scale in the batch norm.
    batch_norm_fused: bool scalar, whether to use the fused batch norm.

  Returns:
    a callable that produces input to slim.arg_scope for specifying argument
    scope of slim.conv2d op.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'center': batch_norm_center,
      'scale': batch_norm_scale,
      'fused': batch_norm_fused}

  def arg_scope_fn():
    with slim.arg_scope([slim.conv2d],
        weights_regularizer=slim.l2_regularizer(scale=weight_decay),
        weights_initializer=slim.xavier_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params) as sc:
      return sc
  return arg_scope_fn


def unit_fn_v2(inputs,
               depth,
               stride,
               shortcut_connection,
               shortcut_from_preact,
               scope=None):
  """The v2 variant of Resnet unit with batch_norm-relu preceeding convolution.
 
  Args:
    inputs: float tensor with shape [batch, height, width, channels], the input
      feature map.
    depth: int scalar, the depth of the two conv ops in each Resnet unit.
    stride: int scalar, the stride of the first conv op in each Resnet unit.
    shortcut_connection: bool scalar, whether to add shortcut connection
      in each Resnet unit. If False, degenerates to a 'Plain network'.
    shortcut_from_preact: bool scalar, whether the shortcut connection starts
      from the preactivation or the input feature map.
    scope: string scalar, scope name of unit.

  Returns:
    output: float tensor with shape [batch, height, width, channels], the 
      output feature map. 
  """
  with tf.variable_scope(scope, 'unit_fn_v2', [inputs]):
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(
        inputs, activation_fn=tf.nn.relu, scope='preact')

    shortcut = preact if shortcut_from_preact else inputs

    if depth != depth_in:
      with tf.name_scope('shortcut'):
        shortcut = slim.avg_pool2d(shortcut, [2, 2], stride=2)
        shortcut = tf.pad(
            shortcut, [[0, 0], [0, 0], [0, 0], [(depth - depth_in) // 2] * 2])

    residual = slim.conv2d(preact, depth, 3, stride, scope='conv1')
    residual = slim.conv2d(residual, depth, 3, 1,
        normalizer_fn=None, activation_fn=None, biases_initializer=None,
        scope='conv2')

    output = residual + shortcut if shortcut_connection else residual

    return output


def resnet_v2_block(scope,
                    num_units,
                    depth,
                    stride,
                    shortcut_connection,
                    shortcut_from_preact):
  """Helper for creating Resnet v2 block specifications.

  Args:
    scope: string scalar, scope name of block.
    num_units: int scalar, the num of Resnet units in each block.
    depth: int scalar, the depth of the two conv ops in each Resnet unit.
    stride: int scalar, the stride of the first conv op in each Resnet unit.
    shortcut_connection: bool scalar, whether to add shortcut connection
      in each Resnet unit. If False, degenerates to a 'Plain network'.
    shortcut_from_preact: bool scalar, whether the shortcut connection starts
      from the preactivation or the input feature map.

  Returns:
    a `Block` instance containing specs for each Resnet block.
  """
  args = [{'depth': depth,
           'stride': stride if i == 0 else 1,
           'shortcut_connection': shortcut_connection,
           'shortcut_from_preact': shortcut_from_preact if i == 0 else False}
      for i in range(num_units)]
  return Block(scope, unit_fn_v2, args)

