"""Defines ResNetModel `Trainer` and `Evaluator` for 
 training on the training set (50000 labeles images) and evaluating
 on the test set (10000 labeled images).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import data


class _BaseModelRunner(object):
  """Base class for `Trainer` and `Evaluator`."""
  mode = None
  def __init__(self, builder, hparams):
    tf.contrib.learn.ModeKeys.validate(type(self).mode)
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._dataset = data.CIFAR10Dataset(hparams, type(self).mode)
      self._model = builder(hparams, self.dataset, type(self).mode)

      self._loss = _compute_loss(self.dataset._labels, self.model._logits)
      self._accuracy = _compute_acc(self.dataset._labels, self.model._logits)

      if type(self).mode == tf.contrib.learn.ModeKeys.TRAIN:
        self._global_step, self._learning_rate = self._get_learning_rate_ops(
          hparams)
        self._update_op = self._get_update_op(hparams)

      self._global_variables_initializer = tf.global_variables_initializer()
      self._saver = tf.train.Saver(
          tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

  @property
  def graph(self):
    return self._graph

  @property
  def dataset(self):
    return self._dataset

  @property
  def model(self):
    return self._model

  def restore_params_from_dir(self, sess, ckpt_dir):
    """Loads parameters from the most up-to-date checkpoint file in
    `ckpt_dir`, or creating new parameters."""
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if latest_ckpt:
      self.restore_params_from_ckpt(sess, latest_ckpt)
    else:
      print("%s model is creating fresh params..." %
          type(self).mode.upper())
      sess.run(self._global_variables_initializer)

  def restore_params_from_ckpt(self, sess, ckpt):
    """Loads parameters from checkpoint `ckpt`."""
    print("%s model is loading params from %s..." % (
        type(self).mode.upper(), ckpt))
    self._saver.restore(sess, ckpt)

  def persist_params_to(self, sess, ckpt):
    """Saves parameters to a checkpoint."""
    print("%s model is saving params to %s..." % (
        type(self).mode.upper(), ckpt))
    self._saver.save(sess, ckpt, global_step=self._global_step)


class ResNetModelTrainer(_BaseModelRunner):
  """Defines `Trainer`, where the graph for learning rate and update ops are 
   defined on top of existing forward pass graph.
  """
  mode = tf.contrib.learn.ModeKeys.TRAIN
  def __init__(self, builder, hparams):
    super(ResNetModelTrainer, self).__init__(
        builder=builder, hparams=hparams)
    with self.graph.as_default():
      self._summary = tf.summary.merge([
          tf.summary.scalar("train_loss", self._loss),
          tf.summary.scalar("train_accuracy", self._accuracy),
          tf.summary.scalar("learning_rate", self._learning_rate)])

  def _get_learning_rate_ops(self, hparams, scope=None):
    with tf.variable_scope(scope, "learning_rate_ops"):
      global_step = tf.Variable(hparams.init_global_step,
          trainable=False, name="global_step")

      init_lr = hparams.learning_rate
      learning_rate = tf.cond(tf.less(global_step, 500),
          lambda: tf.constant(init_lr/10., dtype=tf.float32),
          lambda: tf.cond(tf.less(global_step, 32000),
              lambda: tf.constant(init_lr, dtype=tf.float32),
              lambda: tf.cond(tf.less(global_step, 48000),
                  lambda: tf.constant(init_lr/10., dtype=tf.float32), 
                  lambda: tf.constant(init_lr/100., dtype=tf.float32))))
    return global_step, learning_rate

  def _get_update_ops(self, hparams, scope=None):
    with tf.variable_scope(scope, "update_ops"): 
      opt = tf.train.MomentumOptimizer(self._learning_rate, hparams.momentum)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      # For updating the population mean and variance in batch norm
      with tf.control_dependencies(update_ops):
        update_op = opt.minimize(self._loss, global_step=self._global_step)
    return update_op

  def train(self, sess):
    feed_dict = self.dataset.refill_feed_dict()
    return sess.run([self._update_op,
                     self._loss,
                     self._accuracy,
                     self._learning_rate,
                     self._global_step,
                     self._summary], feed_dict)


class ResNetModelEvaluator(_BaseModelRunner):
  """Defines `Evaluator`."""
  mode = tf.contrib.learn.ModeKeys.EVAL
  def __init__(self, builder, hparams):
    super(ResNetModelEvaluator, self).__init__(
        builder=builder, hparams=hparams)
    with self.graph.as_default():
      self._summary = tf.summary.merge([
          tf.summary.scalar("loss", self._loss),
          tf.summary.scalar("accuracy", self._accuracy)])

  def eval(self, sess):
    feed_dict = self.dataset.refill_feed_dict()
    return sess.run([self._loss,
                     self._accuracy,
                     self._summary], feed_dict)


def _compute_loss(labels, logits, scope=None):
  with tf.variable_scope(scope, "compute_loss", values=[labels, logits]):
    xentropy_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits))
    regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([xentropy_loss] + regularization_loss)
  return total_loss


def _compute_acc(labels, logits, scope=None):
  with tf.variable_scope(scope, "compute_accuracy", values=[labels, logits]):
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(labels, tf.argmax(logits, 1)), tf.float32))
  return accuracy
