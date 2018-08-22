r"""Executable for training Resnet model for classifying Cifar10 images.
  
You need to specify the required flag: `path` -- the path to the directory 
containing Cifar10 binary files. You can leave other flags as default or
make changes.

Example:
  python run_trainer.py \
    --path=/PATH/TO/CIFAR10/ \
    --num_layers=110
"""
import tensorflow as tf

import resnet_model
from utils import resnet_utils
from utils import model_utils
from dataset import TrainerCifar10Dataset


flags = tf.app.flags

flags.DEFINE_string('path', '', 'The path to the directory containing Cifar10'
                    ' binary files.')
flags.DEFINE_string('ckpt_path', '/tmp/resnet/model', 'The path to the ' 
                    'directory to save checkpoint files.')
flags.DEFINE_string('train_log_path', '.', 'The path to the directory to save ' 
                    'summaries.')
flags.DEFINE_integer('num_layers', 20, 'Number of weighted layers. Valid ' 
                     'values: 20, 32, 44, 56, 110')
flags.DEFINE_integer('batch_size', 128, 'Batch size for training.')
flags.DEFINE_integer('epochs', 64000, 'The number of epochs.')
flags.DEFINE_integer('log_per_steps', 200, 'Every N steps to persist model and' 
                     ' print evaluation metrics')
flags.DEFINE_integer('buffer_size', 50000, 'The buffer size for shuffling the '
                     'training set.')
flags.DEFINE_integer('max_ckpts', 5, 'Maximum num of ckpts to keep. If '
                     'negative, keep all.')
flags.DEFINE_float('momentum', 0.9, 'Momentum for the momentum optimizer.')
flags.DEFINE_float('init_lr', 0.1, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 2e-4, 'Weight decay for l2 regularization.')
flags.DEFINE_float('batch_norm_decay', 0.99, 'Moving average decay.')
flags.DEFINE_boolean('shortcut_connection', True, 'Whether to add shortcut '
                     'connection. Defaults to True. False for Plain network.')

FLAGS = flags.FLAGS


def main(_):
  assert FLAGS.path, '`path` is missing.'
  conv_hyperparams_fn = resnet_utils.build_arg_scope_fn(
      weight_decay=FLAGS.weight_decay,
      batch_norm_decay=FLAGS.batch_norm_decay)

  model_trainer = resnet_model.ResnetModelTrainer(
      conv_hyperparams_fn, FLAGS.num_layers, FLAGS.shortcut_connection)

  optimizer, learning_rate = model_utils.build_optimizer(
      FLAGS.init_lr, FLAGS.momentum)

  dataset = TrainerCifar10Dataset(FLAGS.batch_size, 4, 32, FLAGS.buffer_size)

  grouped_update_op, total_loss, accuracy, summary, _ = model_trainer.train(
      FLAGS.path, dataset, optimizer, learning_rate)

  persist_saver = model_utils.create_persist_saver(max_to_keep=FLAGS.max_ckpts
      if FLAGS.max_ckpts > 0 else None)

  summary_writer = tf.summary.FileWriter(FLAGS.train_log_path)

  initializers = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(initializers)

  for gs in range(FLAGS.epochs):
    _, total_loss_val, accuracy_val, summary_val = sess.run(
        [grouped_update_op, total_loss, accuracy, summary])
    summary_writer.add_summary(summary_val, gs)
    if gs == 0 or gs % FLAGS.log_per_steps == 0:
      print('loss: %.6g, train accuracy (mini-batch): %.6g, global_step: %g' % (
          total_loss_val,
          accuracy_val,
          gs))
      persist_saver.save(sess, FLAGS.ckpt_path, global_step=gs)
  persist_saver.save(sess, FLAGS.ckpt_path, global_step=gs)

  sess.close()

if __name__ == '__main__':
  tf.app.run()

