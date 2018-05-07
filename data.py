"""The data module for reading, processing, and feeding labeled examples
to the model builder module.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

BYTES_PER_INSTANCE = 3073
HEIGHT, WIDTH, CHANNELS = 32, 32, 3
TRAIN_MODE = tf.contrib.learn.ModeKeys.TRAIN


class CIFAR10Dataset(object):
  """Wrapper for `labels` (class labels) and `inputs` (images) to `Trainer`
  and `Evaluator`.
  """
  def __init__(self, hparams, mode):
    self._mode = mode
    self._labels = tf.placeholder(tf.int64, shape=[None], name="labels")
    self._inputs = tf.placeholder(tf.float32, shape=[None,
        hparams.height, hparams.width, hparams.channels], name="inputs")

    train, test = _DataReader(hparams)._read_data()
    train, test = _subtract_per_pixel_mean(train, test)

    if mode == TRAIN_MODE:
      self.generator = _DataFeeder(
          hparams, mode, train[0], train[1]).create_train_batch_generator()
    else:
      self.generator = _DataFeeder(
          hparams, mode, test[0], test[1]).create_test_batch_generator()

  @property
  def mode(self):
    return self._mode

  @property
  def labels(self):
    return self._labels

  @property
  def inputs(self):
    return self._inputs

  def refill_feed_dict(self):
    labels, inputs = self.generator.next() 
    return {self.labels: labels, self.inputs: inputs}


class _DataReader(object):
  """Helper class for reading CIFAR10 data."""
  def __init__(self, hparams):
    self.path = hparams.path
    self.train_files = hparams.train_files
    self.test_files = hparams.test_files
    self.bytes_per_instance = BYTES_PER_INSTANCE
    self.channels = CHANNELS
    self.height = HEIGHT 
    self.width = WIDTH

  def _read_data(self):
    filenames = [os.path.join(self.path, fn)
        for fn in self.train_files + self.test_files]

    train = [[], []]
    for fn in filenames[:-1]:
      labels, inputs = self._read_file(fn)
      train[0].append(labels)
      train[1].append(inputs)

    train[0] = np.hstack(train[0])
    train[1] = np.vstack(train[1])
    test = self._read_file(filenames[-1])
    return train, test

  def _read_file(self, filename):
    content = np.fromfile(filename, np.uint8)
    labels = []
    inputs = []
    for i in np.arange(0, len(content), self.bytes_per_instance):
      labels.append(content[i])
      inputs.append(content[i + 1 : i + self.bytes_per_instance].reshape((
          self.channels, self.height, self.width)))
    labels = np.array(labels).astype(np.int64)
    inputs = np.array(inputs).transpose(0, 2, 3, 1).astype(np.float32)
    return [labels, inputs]


class _DataFeeder(object):
  """Helper class for processing and feeding CIFAR10 data."""
  def __init__(self, hparams, mode, labels, inputs):
    self.pad_size = hparams.pad_size
    self.crop_size = hparams.crop_size
    self._rs = np.random.RandomState(hparams.random_seed)
    self.batch_size = hparams.batch_size if mode == TRAIN_MODE \
        else hparams.test_batch_size

    self.labels = labels
    self.inputs = inputs

  def _random_crop(self, image):
    h_crop, w_crop = [self.crop_size] * 2
    h_img, w_img = image.shape[:2]
    h = self._rs.randint(0, h_img - h_crop + 1)
    w = self._rs.randint(0, w_img - w_crop + 1)
    return image[h:h+h_crop, w:w+w_crop]

  def _random_flip_left_right(self, image):
    if self._rs.binomial(1, .5) == 1:
      image = image[:, ::-1]
    return image

  def _pad_image(self, image):
    pad_height, pad_width = [self.pad_size] * 2
    pad_image = np.zeros((image.shape[0] + 2 * pad_height,
                          image.shape[1] + 2 * pad_width, image.shape[2]))
    pad_image[pad_height:-pad_height, pad_width:-pad_width] = image
    return pad_image

  def _preprocess_train_image(self, image):
    image = self._pad_image(image)
    image = self._random_flip_left_right(image)
    image = self._random_crop(image)
    return image[np.newaxis, :]

  def create_train_batch_generator(self):
    """Creates python generator for `Trainer`."""
    while True:
      indices = self._rs.choice(self.inputs.shape[0], self.batch_size, False)
      batch = self.inputs[indices]
      batch = np.vstack([self._preprocess_train_image(img) for img in batch])
      yield self.labels[indices], batch

  def create_test_batch_generator(self):
    """Creates python generator for `Evaluator`."""
    for i in range(0, self.inputs.shape[0], self.batch_size):
      yield self.labels[i: i + self.batch_size], \
          self.inputs[i: i + self.batch_size]


def _subtract_per_pixel_mean(train, test):
  """Computes per pixel means of training set, and subtract them from
  training set and test set."""
  height, width, channels = HEIGHT, WIDTH, CHANNELS
  train[1] = train[1].reshape((-1, height * width * channels))
  test[1] = test[1].reshape((-1, height * width * channels))
  scaler = StandardScaler(with_mean=True, with_std=False)
  train[1] = scaler.fit_transform(train[1]).reshape((
      -1, height, width, channels))
  test[1] = scaler.transform(test[1]).reshape((
      -1, height, width, channels))
  return train, test
