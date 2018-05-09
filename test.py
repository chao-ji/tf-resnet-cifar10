"""Evaluate ResNets on CIFAR10 dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import tensorflow as tf

import data
import model
import model_runners
from utils import arg_utils

tf.logging.set_verbosity(tf.logging.WARN)
FLAGS = None


def test_fn(unused_argv):
  hparams = arg_utils.create_hparams(FLAGS)
  print("#hparams:")
  arg_utils.print_args(hparams)
  print()

  ckpt_dir = hparams.ckpt_dir

  print("Creating evaluator...")
  evaluator = model_runners.ResNetModelEvaluator(model.ResNetModel, hparams)
  print("Done creating evaluator!\n")

  eval_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True),
      graph=evaluator.graph)
 
  print("Start evaluation...")
  for i in range(60000, 64001, 200):
    evaluator.restore_params_from_ckpt(
        eval_sess, os.path.join(ckpt_dir, "model-%d" % i))
    loss, accuracy, summary = evaluator.eval(eval_sess)
    print("accuracy: %f" % accuracy)   
  print("Done evaluation!")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  arg_utils.add_arguments(parser)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=test_fn, argv=[sys.argv[0]] + unparsed)
