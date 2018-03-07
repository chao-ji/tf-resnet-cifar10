import tensorflow as tf

import data

class _BaseModelRunner(object):
  mode = None
  def __init__(self, builder, hparams):
    tf.contrib.learn.ModeKeys.validate(type(self).mode)
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._dataset = data.CIFAR10Dataset(hparams, type(self).mode)
      self._model = builder(hparams, self.dataset, type(self).mode)
      self._accuracy = self._compute_accuracy()

      if type(self).mode == tf.contrib.learn.ModeKeys.TRAIN:
        self._global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learning_rate = self._get_learning_rate(hparams)
        self.update_op = self._get_update_op(hparams)

      self._global_variables_initializer = tf.global_variables_initializer()
      self._params = tf.global_variables()
      self._saver = tf.train.Saver(
          self._params, max_to_keep=hparams.num_keep_ckpts)

  @property
  def graph(self):
    return self._graph

  @property
  def dataset(self):
    return self._dataset

  @property
  def model(self):
    return self._model

  def restore_params_from(self, sess, ckpt_dir):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if latest_ckpt:
      print("%s model is loading params from %s..." % (
          type(self).mode.upper(), latest_ckpt))
      self._saver.restore(sess, latest_ckpt)
    else:
      print("%s model is creating fresh params..." %
          type(self).mode.upper())
      sess.run(self._global_variables_initializer)

  def persist_params_to(self, sess, ckpt):
    print("%s model is saving params to %s..." % (
        type(self).mode.upper(), ckpt))
    self._saver.save(sess, ckpt, global_step=self._global_step)

  def _compute_accuracy(self):
    logits = self.model.logits
    labels = self.dataset.labels
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(labels, tf.argmax(logits, 1)), tf.float32))
    return accuracy


class ResNetModelTrainer(_BaseModelRunner):
  mode = tf.contrib.learn.ModeKeys.TRAIN
  def __init__(self, builder, hparams):
    super(ResNetModelTrainer, self).__init__(
        builder=builder, hparams=hparams)
    with self.graph.as_default():
      self.summary = tf.summary.merge([
          tf.summary.scalar("train_loss", self.model.loss),
          tf.summary.scalar("train_accuracy", self._accuracy),
          tf.summary.scalar("learning_rate", self.learning_rate)])

  def _get_learning_rate(self, hparams):
    init_lr = hparams.learning_rate
    global_step = self._global_step
    learning_rate = tf.cond(tf.less(global_step, 500),
        lambda: tf.constant(0.01, dtype=tf.float32),
        lambda: tf.cond(tf.less(global_step, 32000),
            lambda: tf.constant(init_lr, dtype=tf.float32),
            lambda: tf.cond(tf.less(global_step, 48000),
                lambda: tf.constant(init_lr/10., dtype=tf.float32), 
                lambda: tf.constant(init_lr/100., dtype=tf.float32))))
    return learning_rate

  def _get_update_op(self, hparams):
    opt = tf.train.MomentumOptimizer(self.learning_rate, hparams.momentum)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      update_op = opt.minimize(self.model.loss, global_step=self._global_step)
    return update_op

  def train(self, sess):
    feed_dict = self.dataset.refill_feed_dict()
    return sess.run([self.update_op,
                     self.model.loss,
                     self._accuracy,
                     self.model.batch_size,
                     self.learning_rate,
                     self._global_step,
                     self.summary], feed_dict)

class ResNetModelEvaluator(_BaseModelRunner):
  mode = tf.contrib.learn.ModeKeys.EVAL
  def __init__(self, builder, hparams):
    super(ResNetModelEvaluator, self).__init__(
        builder=builder, hparams=hparams)
    with self.graph.as_default():
      self.summary = tf.summary.merge([
          tf.summary.scalar("loss", self.model.loss),
          tf.summary.scalar("accuracy", self._accuracy)])

  def eval(self, sess):
    feed_dict = self.dataset.refill_feed_dict()
    return sess.run([self.model.loss,
                     self._accuracy,
                     self.model.batch_size,
                     self.summary], feed_dict)
