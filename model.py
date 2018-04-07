import tensorflow as tf
from tensorflow.python.layers import layers

import data

BatchNorm = layers.BatchNormalization
Conv = layers.Conv2D
Dense = layers.Dense
AvgPool = layers.AveragePooling2D

class ResNetModel(object):
  def __init__(self, hparams, dataset, mode, residual_connection=True):
    self._mode = mode
    self._filters = hparams.num_filters 
    self._num_blocks = hparams.num_blocks
    self._training = (self._mode == tf.contrib.learn.ModeKeys.TRAIN)
    self._residual_connection = residual_connection
    self._batch_size = tf.size(dataset.labels)
    self.logits = self._build_graph(hparams, dataset)

  @property
  def mode(self):
    return self._mode

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def residual_connection(self):
    return self._residual_connection

  def _build_graph(self, hparams, dataset):
    inputs = dataset.images

    with tf.variable_scope("Pre_Residual_Layer"):
      ftmps = self._conv_bn_relu(inputs, self._filters[0], (1, 1), hparams)
       
    for i in range(self._num_blocks):
      with tf.variable_scope("Filters_%d_Block_%d" % (self._filters[0], i)):
        ftmps = self._residual_block(
            ftmps, self._filters[0], self._filters[0], hparams, i == 0)

    for i in range(self._num_blocks):
      with tf.variable_scope("Filters_%d_Block_%d" % (self._filters[1], i)):
        ftmps = self._residual_block(
            ftmps, self._filters[0] if i == 0 else self._filters[1],
            self._filters[1], hparams, False)

    for i in range(self._num_blocks):
      with tf.variable_scope("Filters_%d_Block_%d" % (self._filters[2], i)):
        ftmps = self._residual_block(
            ftmps, self._filters[1] if i == 0 else self._filters[2],
            self._filters[2], hparams, False)

    with tf.variable_scope("Post_Residual_Layer"):
      logits = self._final_block(ftmps, hparams.num_classes, hparams)

    return logits

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

    ftmps = conv_layer(inputs)   
    ftmps = batch_norm_layer(ftmps, training=self._training)
    ftmps = tf.nn.relu(ftmps)
    return ftmps

  def _residual_block(self, inputs, in_depth, out_depth, hparams, first_block):
    ftmps = inputs 
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
        ftmps = batch_norm_layers[0](ftmps, training=self._training)
        ftmps = tf.nn.relu(ftmps)
      ftmps = conv_layers[0](ftmps)

    with tf.variable_scope("ConvLayer2"):
      ftmps = batch_norm_layers[1](ftmps, training=self._training)
      ftmps = tf.nn.relu(ftmps)
      ftmps = conv_layers[1](ftmps)

    if self.residual_connection:
      if in_depth == out_depth:
        padded_inputs = inputs
      else:
        pooled_inputs = avg_pool_layer(inputs)
        padded_inputs = tf.pad(pooled_inputs,
            [[0, 0], [0, 0], [0, 0], [(out_depth - in_depth) / 2] * 2])
      return ftmps + padded_inputs
    else:
      return ftmps

  def _final_block(self, inputs, num_units, hparams):
    batch_norm_layer = BatchNorm(axis=-1,
                                 fused=False,
                                 epsilon=hparams.epsilon)
    dense_layer = Dense(num_units,
                        use_bias=True,
                        kernel_initializer=_dense_kernel_initializer(hparams),
                        kernel_regularizer=_regularizer(hparams))

    ftmps = batch_norm_layer(inputs, training=self._training)
    ftmps = tf.nn.relu(ftmps)
    ftmps_global_avg_pool = tf.reduce_mean(ftmps, reduction_indices=[1, 2])
    logits = dense_layer(ftmps_global_avg_pool)
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
