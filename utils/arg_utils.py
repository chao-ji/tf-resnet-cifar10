"""Utilities for handling argument parsing and hyperparameters.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def add_arguments(parser):
  parser.register("type", "bool", lambda v: v.lower() == "true")

  parser.add_argument("--path", type=str, required=True,
      help="Input dir containing cifar10 dataset")
  parser.add_argument("--ckpt_dir", type=str, default="/tmp/resnet",
      help="Output dir where checkpoint files are saved")
  parser.add_argument("--shortcut_conn", type="bool", nargs="?", const=True,
      default=True, help=("Whether to add shortcut connection:" +
      "True for ResNets, False for PlainNets"))
  parser.add_argument("--fused", type="bool", nargs="?", const=True,
      default=True, help="Whether to use fused batch normalization")
  parser.add_argument("--weight_decay", type=float, default=2e-4,
      help="Weight decay")
  parser.add_argument("--epsilon", type=float, default=1e-3,
      help="Epsilon of batch normalization")
  parser.add_argument("--momentum", type=float, default=0.9,
      help="Momentum of SGD")
  parser.add_argument("--learning_rate", type=float, default=0.1,
      help="Initial learning rate")
  parser.add_argument("--num_layers", type=int, default=20,
      help="Num of layers: 20|32|44|56|110")
  parser.add_argument("--init_global_step", type=int, default=0,
      help="Initial global step")
  parser.add_argument("--random_seed", type=int, default=None,
      help="Random seed")
  parser.add_argument("--batch_size", type=int, default=128,
      help="Training Batch size")
  parser.add_argument("--log_per_steps", type=int, default=200,
      help="Every N steps to persist model and print evaluation metrics")
  parser.add_argument("--num_keep_ckpts", type=int, default=None,
      help="Num of most recent checkpoints to keep:" +
      "Defaults to unlimited checkpoints")


def create_hparams(FLAGS):
  hparams = tf.contrib.training.HParams(
      crop_size=32,
      pad_size=4,
      **vars(FLAGS))
  return hparams


def print_args(hparams):
  for k, v in hparams.values().iteritems():
    print("%s: %s" % (k, v)) 
