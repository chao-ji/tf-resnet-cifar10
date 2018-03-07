import tensorflow as tf
from tensorflow.python.layers import layers

import data

BatchNorm = layers.BatchNormalization
Conv = layers.Conv2D
Dense = layers.Dense
AvgPool = layers.AveragePooling2D

class ResNetModel(object):
  def __init__(self, hparams, dataset, mode, scope=None):
    self._mode = mode
    self._filters = hparams.num_filters 
    self._num_blocks = hparams.num_blocks
    self._training = (self._mode == tf.contrib.learn.ModeKeys.TRAIN)

    self._batch_size = tf.size(dataset.labels)
    self.logits, self.loss = self._build_graph(hparams, dataset, scope)

  @property
  def mode(self):
    return self._mode

  @property
  def batch_size(self):
    return self._batch_size

  def _build_graph(self, hparams, dataset, scope=None):
    inputs = dataset.images

    with tf.variable_scope("Pre_Residual_Layer"):
      outputs = self._conv_bn_relu(inputs, self._filters[0], (1, 1), hparams)
       
    for i in range(self._num_blocks):
      with tf.variable_scope("Filters_%d_Block_%d" % (self._filters[0], i)):
        outputs = self._residual_block(
            outputs, self._filters[0], self._filters[0], hparams, i == 0)

    for i in range(self._num_blocks):
      with tf.variable_scope("Filters_%d_Block_%d" % (self._filters[1], i)):
        outputs = self._residual_block(
            outputs, self._filters[0] if i == 0 else self._filters[1],
            self._filters[1], hparams, False)

    for i in range(self._num_blocks):
      with tf.variable_scope("Filters_%d_Block_%d" % (self._filters[2], i)):
        outputs = self._residual_block(
            outputs, self._filters[1] if i == 0 else self._filters[2],
            self._filters[2], hparams, False)

    with tf.variable_scope("Post_Residual_Layer"):
      logits = self._final_block(outputs, hparams.num_classes, hparams)

    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      loss = _compute_loss(logits, dataset.labels)
    else:
      loss = tf.no_op()
    return logits, loss

  def _conv_bn_relu(self, inputs, filters, strides, hparams):
    conv_layer = Conv(filters=filters,
                      kernel_size=hparams.kernel_size,
                      strides=strides,
                      padding="SAME",
                      use_bias=False,
                      kernel_initializer=_conv_kernel_initializer(hparams),
                      kernel_regularizer=_regularizer(hparams))
    batch_norm_layer = BatchNorm(axis=-1,
                                 fused=False,
                                 epsilon=hparams.epsilon)

    outputs = conv_layer(inputs)   
    outputs = batch_norm_layer(outputs, training=self._training)
    outputs = tf.nn.relu(outputs)
    return outputs

  def _residual_block(self, inputs, in_depth, out_depth, hparams, first_block):
    outputs = inputs 
    batch_norm_layers = [BatchNorm(axis=-1,
                                   fused=False,
                                   epsilon=hparams.epsilon),
                         BatchNorm(axis=-1,
                                   fused=False,
                                   epsilon=hparams.epsilon)]
    conv_layers = [Conv(filters=out_depth,
                        kernel_size=hparams.kernel_size,
                        strides=(1, 1) if in_depth == out_depth else (2, 2),
                        padding="SAME",
                        use_bias=False,
                        kernel_initializer=_conv_kernel_initializer(hparams),
                        kernel_regularizer=_regularizer(hparams)),
                   Conv(filters=out_depth,
                        kernel_size=hparams.kernel_size,
                        strides=(1, 1),
                        padding="SAME",
                        use_bias=False,
                        kernel_initializer=_conv_kernel_initializer(hparams),
                        kernel_regularizer=_regularizer(hparams))]
    avg_pool_layer = AvgPool(pool_size=(2, 2),
                             strides=(2, 2),
                             padding="VALID") 

    with tf.variable_scope("ConvLayer1"):
      if not first_block:
        outputs = batch_norm_layers[0](outputs, training=self._training)
        outputs = tf.nn.relu(outputs)
      outputs = conv_layers[0](outputs) 

    with tf.variable_scope("ConvLayer2"):
      outputs = batch_norm_layers[1](outputs, training=self._training)
      outputs = tf.nn.relu(outputs)
      outputs = conv_layers[1](outputs)

    if in_depth == out_depth:
      padded_inputs = inputs
    else:
      pooled_inputs = avg_pool_layer(inputs)
      padded_inputs = tf.pad(pooled_inputs,
          [[0, 0], [0, 0], [0, 0], [(out_depth - in_depth) / 2] * 2])

    return outputs + padded_inputs

  def _final_block(self, inputs, num_units, hparams):
    batch_norm_layer = BatchNorm(axis=-1,
                                 fused=False,
                                 epsilon=hparams.epsilon)
    dense_layer = Dense(num_units,
                        use_bias=True,
                        kernel_initializer=_dense_kernel_initializer(hparams),
                        kernel_regularizer=_regularizer(hparams))

    outputs = batch_norm_layer(inputs, training=self._training)
    outputs = tf.nn.relu(outputs)
    outputs_global_avg_pool = tf.reduce_mean(outputs, reduction_indices=[1, 2])
    logits = dense_layer(outputs_global_avg_pool)
    return logits

def _compute_loss(logits, labels):
  xentropy_loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits))
  regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n([xentropy_loss] + regularization_loss)
  return total_loss

def _conv_kernel_initializer(hparams):
  return tf.contrib.layers.xavier_initializer(seed=hparams.random_seed)

def _dense_kernel_initializer(hparams):
  return tf.initializers.variance_scaling(
      distribution="uniform", seed=hparams.random_seed)

def _regularizer(hparams):
  return tf.contrib.layers.l2_regularizer(scale=hparams.weight_decay)
