"""Train ResNets on CIFAR10 dataset."""
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
EPOCHS = 64000
FLAGS = None


def train_fn(unused_argv):
  hparams = arg_utils.create_hparams(FLAGS)
  print("#hparams:")
  arg_utils.print_args(hparams)
  print()

  ckpt_dir = hparams.ckpt_dir 

  print("Creating trainer...") 
  trainer = model_runners.ResNetModelTrainer(model.ResNetModel, hparams)
  print("Done creating trainer!\n")

  train_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True),
      graph=trainer.graph)
  trainer.restore_params_from_dir(train_sess, ckpt_dir)

  summary_writer = tf.summary.FileWriter(os.path.join(ckpt_dir, "train_log"))

  print("\nStart training...")
  for i in range(EPOCHS):
    if i % hparams.log_per_steps == 0:
      trainer.persist_params_to(train_sess, os.path.join(ckpt_dir, "model"))

    _, loss, accuracy, lr, global_step, summary = trainer.train(train_sess)
    summary_writer.add_summary(summary, global_step)

    if i % hparams.log_per_steps == 0:
      print("step %d, training accuracy %g, loss %g, lr %g" %
          (i, accuracy, loss, lr))

  trainer.persist_params_to(train_sess, os.path.join(ckpt_dir, "model"))
  print("Done training!")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  arg_utils.add_arguments(parser)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=train_fn, argv=[sys.argv[0]] + unparsed)
