import tensorflow as tf
import os

PREFIX = 'resnet-cifar10'


class ResNetCifar10Trainer(object):
  """Trains a ResNetCifar10 model."""
  def __init__(self, model):
    """Constructor.

    Args:
      model: an instance of ResNetCifar10Model instance.
    """
    self._model = model

  def train(self,
            dataset, 
            optimizer, 
            ckpt, 
            batch_size, 
            num_iterations, 
            log_per_iterations, 
            ckpt_path,
            logdir='log'):
    """Executes training.

    Args:
      dataset: a tf.data.Dataset instance, the input data generator.
      optimizer: a tf.keras.optimizers.Optimizer instance, applies gradient 
        updates.
      ckpt: a tf.train.Checkpoint instance, saves or load weights.
      batch_size: int scalar, batch size.
      num_iterations: int scalar, num of iterations to train the model.
      log_per_iterations: int scalar, saves weights to checkpoints every 
        `log_per_iterations` iterations.
      ckpt_path: string scalar, the path to the directory that the checkpoint 
        files will be written to or loaded from. 
      logdir: string scalar, the directory that the tensorboard log data will
        be written to.
    """
    train_step_signature = [
        tf.TensorSpec(shape=(batch_size,), dtype=tf.int64),
        tf.TensorSpec(shape=(batch_size, 32, 32, 3), dtype=tf.float32)]

    @tf.function(input_signature=train_step_signature)
    def train_step(labels, images):
      with tf.GradientTape() as tape:
        logits = self._model(images, training=True)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits))
        regularization_losses = self._model.losses
        total_loss = tf.add_n(regularization_losses + [cross_entropy_loss])

      accuracy = tf.reduce_mean(tf.cast(tf.equal(
          labels, tf.argmax(logits, 1)), 'float32'))
      gradients = tape.gradient(total_loss, self._model.trainable_variables)

      optimizer.apply_gradients(
          zip(gradients, self._model.trainable_variables))
      step = optimizer.iterations
      lr = optimizer.learning_rate(step)
      
      return total_loss, accuracy, step - 1, lr

    summary_writer = tf.summary.create_file_writer(logdir)

    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
      print('Restoring from checkpoint: %s ...' % latest_ckpt)
      ckpt.restore(latest_ckpt)
    else:
      print('Training from scratch...')

    for labels, images in dataset:
      total_loss, accuracy, step, lr = train_step(labels, images)

      with summary_writer.as_default():
        tf.summary.scalar('train_loss', total_loss, step=step)
        tf.summary.scalar('train_accuracy', accuracy, step=step)

      if step % log_per_iterations == 0:
        print('global_step: %d, loss: %f, accuracy: %f, lr: %f' % (
            step, total_loss.numpy(), accuracy.numpy(), lr.numpy()))

        ckpt.save(os.path.join(ckpt_path, PREFIX))

      if step == num_iterations:
        break


class ResNetCifar10Evaluator(object):
  """Evaluates a ResNetCifar10 model."""
  def __init__(self, model):
    """Constructor.

    Args:
      model: an instance of ResNetCifar10Model instance.
    """
    self._model = model

  def evaluate(self, dataset, batch_size):
    """Executes evaluation.

    Args:
      dataset: a tf.data.Dataset instance, the input data generator.
      batch_size: int scalar, batch size.

    Returns:
      total_loss_list: a list of floats, the loss evaluated at each of the
        checkpoint.
      accuracy_list: a list of floats, the accuracy evaluted at eac hof the
        checkpoint.
    """
    evaluate_step_signature = [
        tf.TensorSpec(shape=(batch_size,), dtype=tf.int64),
        tf.TensorSpec(shape=(batch_size, 32, 32, 3), dtype=tf.float32)]
 
    @tf.function(input_signature=evaluate_step_signature) 
    def evaluate_step(labels, images):
      logits = self._model(images, training=False)

      cross_entropy_loss = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=labels, logits=logits))

      regularization_losses = self._model.losses

      total_loss = tf.add_n(regularization_losses + [cross_entropy_loss])
      accuracy = tf.reduce_mean(tf.cast(tf.equal(
          labels, tf.argmax(logits, 1)), 'float32'))

      return total_loss, accuracy

    total_loss_list = []
    accuracy_list = []

    for labels, images in dataset:
      total_loss, accuracy = evaluate_step(labels, images)
      total_loss_list.append(total_loss.numpy())
      accuracy_list.append(accuracy.numpy())

    return total_loss_list, accuracy_list


def build_optimizer(init_lr=0.1, momentum=0.9):
  """Builds the optimizer for training.

  Args:
    init_lr: float scalar, initial learning rate.
    momentum: float scalar, momentum for SGD.

  Returns:
    optimizer: an instance of tf.keras.optimizers.SGD.
  """ 
  learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      [500, 32000, 48000], 
      [init_lr / 10., init_lr, init_lr / 10., init_lr / 100.])
  optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=momentum)
  return optimizer  
