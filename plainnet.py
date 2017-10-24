import os
import numpy as np
import tensorflow as tf
from cifar10_bin import *
from sklearn.preprocessing import StandardScaler

class PlainNet(object):
  def __init__(self,
                ckpt_path,
                num_blocks=9,
                filter_size=[3, 3],
                weight_decay=2*1e-4,
                batch_norm_epsilon=1e-3,
                filters=[16, 32, 64],
                seed=123):
    self.ckpt_path = ckpt_path
    self.num_blocks = num_blocks
    self.filter_size = filter_size
    self.weight_decay = weight_decay
    self.batch_norm_epsilon = batch_norm_epsilon
    self.filters = filters
    self.seed = seed
    self.X = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, 3])
    self.Y = tf.placeholder(tf.float32, shape=[None, 10])
    self.SWITCH = tf.placeholder(dtype=tf.float32, shape=[])
    self.LOGITS = None
    self.bn_tensors  = []
    self.population_estimate = {}
    self.sess = tf.InteractiveSession()

  def create_variable(self, name, shape,
                      initializer=tf.truncated_normal_initializer(stddev=0.001),
                      regularize=True):
    regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay) if regularize else None
    variable = tf.get_variable(name=name, shape=shape, initializer=initializer, regularizer=regularizer)
    return variable

  def conv2d(self, X, num_filters, strides):
    W = self.create_variable(name="conv_weight",
                              shape=[self.filter_size[0], self.filter_size[1], int(X.shape[-1]), num_filters],
                              initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
    X = tf.nn.conv2d(X, W, strides=strides, padding="SAME")
    return X

  def bn(self, X):
    offset = self.create_variable(name="bn_offset", shape=[int(X.shape[-1])],
                                  initializer=tf.zeros_initializer(), regularize=False)
    scale = self.create_variable(name="bn_scale", shape=[int(X.shape[-1])],
                                  initializer=tf.ones_initializer(), regularize=False)
    mean_minibatch, var_minibatch = tf.nn.moments(X, axes=[0, 1, 2])
    mean_pop, var_pop = tf.Variable(tf.zeros(shape=[int(X.shape[-1])]), trainable=False), \
                        tf.Variable(tf.ones(shape=[int(X.shape[-1])]), trainable=False)

    SWITCH = self.SWITCH
    mean = mean_minibatch * SWITCH + mean_pop * (1 - SWITCH)
    var = var_minibatch * SWITCH + var_pop * (1 - SWITCH)

    self.bn_tensors.append(X)
    self.population_estimate[X.name] = mean_pop, var_pop

    X = tf.nn.batch_normalization(X, mean, var, offset, scale, self.batch_norm_epsilon)
    return X

  def bn_population_estimate(self, X_batch):
    X, SWITCH = self.X, self.SWITCH
    feed_dict = {X: X_batch, SWITCH: 1.}
    assign_op = []
    for tensor in self.bn_tensors:
      tensor_val = tensor.eval(feed_dict)
      assign_op.append(tf.assign(self.population_estimate[tensor.name][0], tensor_val.mean(axis=(0, 1, 2))))
      assign_op.append(tf.assign(self.population_estimate[tensor.name][1], tensor_val.var(axis=(0, 1, 2))))
    self.sess.run(assign_op)

  def conv_bn_relu(self, X, num_filters, strides):
    X = self.conv2d(X, num_filters, strides)
    X = self.bn(X)
    X = tf.nn.relu(X)
    return X

  def block(self, X, in_channels, out_channels, first_block):
    H = X

    with tf.variable_scope("conv1_in_blk"):
      if not first_block:
        H = self.bn(H)
        H = tf.nn.relu(H)
      H = self.conv2d(H, out_channels, [1, 1, 1, 1] if in_channels == out_channels else [1, 2, 2, 1])

    with tf.variable_scope("conv2_in_blk"):
      H = self.bn(H)
      H = tf.nn.relu(H)
      H = self.conv2d(H, out_channels, [1, 1, 1, 1])







    return H

  def fc_layer(self, X, num_filters):
    W = self.create_variable(name="fc_weight", shape=[int(X.shape[-1]), num_filters],
                              initializer=tf.uniform_unit_scaling_initializer(factor=1.0, seed=self.seed))
    F = tf.matmul(X, W)
    bias = self.create_variable(name="fc_bias", shape=[num_filters], initializer=tf.zeros_initializer())
    F = F + bias
    return F

  def inference(self, X):
    with tf.variable_scope("pre_resblock"):
      H = self.conv_bn_relu(X, self.filters[0], [1, 1, 1, 1])

    for i in range(self.num_blocks):
      with tf.variable_scope("filters_%d_blk_%d" % (self.filters[0], i)):
        H = self.block(H, self.filters[0], self.filters[0], i==0)

    for i in range(self.num_blocks):
      with tf.variable_scope("filters_%d_blk_%d" % (self.filters[1], i)):
        H = self.block(H, self.filters[0] if i == 0 else self.filters[1], self.filters[1], False)

    for i in range(self.num_blocks):
      with tf.variable_scope("filters_%d_blk_%d" % (self.filters[2], i)):
        H = self.block(H, self.filters[1] if i == 0 else self.filters[2], self.filters[2], False)

    with tf.variable_scope("readout_fc"):
      H = self.bn(H)
      H = tf.nn.relu(H)
      H_GLOBAL_AVG_POOL = tf.reduce_mean(H, reduction_indices=[1, 2])
      LOGITS = self.fc_layer(H_GLOBAL_AVG_POOL, 10)

    return LOGITS

  def logits(self, X):
    if self.LOGITS is None:
      self.LOGITS = self.inference(X)
    return self.LOGITS

  def loss(self, Y, LOGITS):
    CROSS_ENTROPY_LOSS = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=LOGITS))
    REG_LOSS = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    TOTAL_LOSS = tf.add_n([CROSS_ENTROPY_LOSS] + REG_LOSS)
    return TOTAL_LOSS

  def accuracy(self, Y, LOGITS):
    CORRECT_PREDICTION = tf.equal(tf.argmax(Y, 1), tf.argmax(LOGITS, 1))
    ACCURACY = tf.reduce_mean(tf.cast(CORRECT_PREDICTION, tf.float32))
    return ACCURACY

  def train(self, X_train, Y_train, data_feeder, global_step, lr, num_steps):
    X, Y, SWITCH = self.X, self.Y, self.SWITCH

    LOGITS = self.logits(X)
    TOTAL_LOSS = self.loss(Y, LOGITS)
    ACCURACY = self.accuracy(Y, LOGITS)

    TRAIN_STEP = tf.train.MomentumOptimizer(lr, 0.9).minimize(TOTAL_LOSS, global_step=global_step)

    self.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

    for i in np.arange(num_steps):
      batch_xs, batch_ys = data_feeder.generate_train_batch(X_train, Y_train)
      feed_dict = {X: batch_xs, Y: batch_ys, SWITCH: 1.0}
      if i % 200 == 0:
        train_accuracy = ACCURACY.eval(feed_dict)
        loss = TOTAL_LOSS.eval(feed_dict)
        print "step %d, training accuracy %g, loss %g" % (i, train_accuracy, loss)
        self.persist(saver, global_step)
      TRAIN_STEP.run(feed_dict)

  def test(self, X_test, batch_size, use_population_estimate=True, from_ckpt=None):
    y_pred = np.array([])
    X, SWITCH = self.X, self.SWITCH
    LOGITS = self.logits(X)
    switch = 0.0 if use_population_estimate else 1.0

    if from_ckpt is not None:
      saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)
      self.sess.run(tf.global_variables_initializer())
      saver.restore(self.sess, from_ckpt)

    for i in range(0, X_test.shape[0], batch_size):
      batch_xs = X_test[i:i+batch_size]
      LOGITS_val = LOGITS.eval({X: batch_xs, SWITCH: switch})
      y_pred = np.append(y_pred, LOGITS_val.argmax(axis=1))

    return y_pred

  def persist(self, saver, global_step):
    saver.save(self.sess, self.ckpt_path, global_step)
