import tensorflow as tf


def compute_loss(labels, logits, scope=None):
  """Computes total loss given groundtruth label and prediction logit.

  Args:
    labels: an int tensor with shape [batch], the groundtruth label.
    logits: a float tensor with shape [batch, 10], the prediction logit.
    scope: string scalar, scope name.

  Returns:
    total_loss: a float scalar tensor, the total loss
  """
  with tf.name_scope('Loss', scope, values=[labels, logits]):
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([cross_entropy_loss] + reg_losses)

    return total_loss


def compute_accuracy(labels, logits, scope=None):
  """Computes accuracy given groundtruth label and prediction logit.

  Args:
    labels: an int tensor with shape [batch], the groundtruth label.
    logits: a float tensor with shape [batch, 10], the prediction logit.
    scope: string scalar, scope name.

  Returns:
    accuracy: a float scalar tensor, the percentage of correct predictions. 
  """
  with tf.name_scope('Accuracy', scope, values=[labels, logits]):
    accuracy = tf.reduce_mean(
        tf.to_float(tf.equal(labels, tf.argmax(logits, 1))))
    return accuracy


def build_optimizer(init_lr, momentum):
  """Build the momentum optimizer and the piecewise constant learning rate.

  Args:
    init_lr: a float scalar, initial learning rate.
    momentum: float scalar, momentum.

  Returns:
    optimizer: an optimizer instance.
    learning_rate: a float scalar tensor, learning rate.
  """
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.cond(tf.less(global_step, 500),
      lambda: tf.constant(init_lr/10., dtype=tf.float32),
      lambda: tf.cond(tf.less(global_step, 32000),
          lambda: tf.constant(init_lr, dtype=tf.float32),
          lambda: tf.cond(tf.less(global_step, 48000),
              lambda: tf.constant(init_lr/10., dtype=tf.float32),
              lambda: tf.constant(init_lr/100., dtype=tf.float32))))

  optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

  return optimizer, learning_rate


def build_training_summary(total_loss, accuracy, learning_rate):
  """Build training summaries.

  Args:
    total_loss: a float scalar tensor, the total loss
    accuracy: a float scalar tensor, the percentage of correct predictions.
    learning_rate: a float scalar tensor, learning rate.

  Returns:
    summary: a scalar summary tensor.
  """
  summary = tf.summary.merge([
          tf.summary.scalar("train_loss", total_loss),
          tf.summary.scalar("train_accuracy", accuracy),
          tf.summary.scalar("learning_rate", learning_rate)])

  return summary


def create_persist_saver(max_to_keep=5):
  """Creates persist saver for persisting variables to a checkpoint file.

  Args:
    max_to_keep: int scalar or None, max num of checkpoints to keep. If None,
      keeps all checkpoints.
        
  Returns:
    persist_saver: a tf.train.Saver instance.
  """
  persist_saver = tf.train.Saver(max_to_keep=max_to_keep)
  return persist_saver


def create_restore_saver():
  """Creates restore saver for persisting variables to a checkpoint file.

  Returns:
    restore_saver: a tf.train.Saver instance.
  """
  restore_saver = tf.train.Saver()
  return restore_saver

