r"""Executable for evaluating a trained Resnet model for classifying Cifar10
images.
  
You need to specify two required flags: `path` -- the path to the directory 
containing Cifar10 binary files, and `ckpt_path` -- the path to the directory
to load trained variables from. You can leave other flags as default or make 
changes.

Example:
  python run_evaluator.py \
    --path=/PATH/TO/CIFAR10 \
    --ckpt_path=/PATH/TO/CKPT \
    --num_layers=110
"""
import tensorflow as tf

import resnet_model
from utils import resnet_utils
from utils import model_utils
from dataset import EvaluatorCifar10Dataset


flags = tf.app.flags

flags.DEFINE_string('path', '', 'The path to the directory containing Cifar10'
                    ' binary files.')
flags.DEFINE_string('ckpt_path', '', 'The path to the checkpoint file to load '
                    'variables from.')
flags.DEFINE_integer('num_layers', 20, 'Number of weighted layers. Valid '
                     'values: 20, 32, 44, 56, 110')
flags.DEFINE_boolean('shortcut_connection', True, 'Whether to add shortcut '
                     'connection. Defaults to True. False for Plain network.')

FLAGS = flags.FLAGS


def main(_):
  assert FLAGS.path, '`path` is missing.'
  assert FLAGS.ckpt_path, '`ckpt_path` is missing.'
  conv_hyperparams_fn = resnet_utils.build_arg_scope_fn()

  model_evaluator = resnet_model.ResnetModelEvaluator(conv_hyperparams_fn, 
                                                      FLAGS.num_layers)

  dataset = EvaluatorCifar10Dataset(batch_size=10000)

  total_loss, accuracy = model_evaluator.evaluate(FLAGS.path, dataset)

  restore_saver = model_utils.create_restore_saver()

  sess = tf.Session()

  restore_saver.restore(sess, FLAGS.ckpt_path)

  loss, acc = sess.run([total_loss, accuracy])
  print('Evaluation loss: %g' % loss)
  print('Evaluation accuracy: %g' % acc)

  sess.close()

if __name__ == '__main__':
  tf.app.run()

