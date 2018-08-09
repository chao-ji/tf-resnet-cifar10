import os
from abc import ABCMeta
from abc import abstractproperty 

import numpy as np
import tensorflow as tf

BYTES_PER_RECORD = 3073
HEIGHT, WIDTH, CHANNELS = 32, 32, 3
EVAL_BATCH_SIZE = 10000
TRAIN_FILES = ('data_batch_1.bin',
               'data_batch_2.bin',
               'data_batch_3.bin',
               'data_batch_4.bin',
               'data_batch_5.bin')
TEST_FILES = 'test_batch.bin'


class Cifar10Dataset(object):
  """An abstract base class to be subclassed by Trainer and Evaluator. All
  methods have been implemented in this base class, except the `mode` property.
  """

  __metaclass__ = ABCMeta

  @abstractproperty
  def mode(self):
    """Returns a string scalar indicating the mode of dataset (train or eval).
    """
    pass

  def get_tensor_dict(self, path):
    """Generates a tensor dict containing labels and images.

    Args:
      path: a string scalar, the path to the directory containing cifar10
        binary files.
    
    Returns:
      tensor_dict: a dict mapping from tensor names to tensors.
        {'labels': int tensor with shape [batch],
         'images': float tensor with shape [batch, height, width, channels]}
    """
    per_pixel_mean = self._get_per_pixel_mean(path)
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      filename_list = [os.path.join(path, fn) for fn in TRAIN_FILES]
    else:
      filename_list = [os.path.join(path, TEST_FILES)]
    dataset = tf.data.FixedLengthRecordDataset(filename_list, BYTES_PER_RECORD)
    dataset = dataset.repeat(None)
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      dataset = dataset.shuffle(buffer_size=self._buffer_size,
                                seed=self._random_seed)

    def transform_fn(record):
      """Transforms a record (string scalar containing a byte sequence) into a 
      tuple of label (int scalar) and image (float tensor with shape 
      [height, width, channels]).
      """
      decoded = tf.decode_raw(record, tf.uint8)
      label, image = decoded[0], decoded[1:]
      label = tf.to_int64(label)
      image = tf.to_float(image) - per_pixel_mean
      image = tf.transpose(
          tf.reshape(image, [CHANNELS, HEIGHT, WIDTH]), [1, 2, 0])
      return label, image
 
    dataset = dataset.map(transform_fn)

    def data_augmentation_fn(*label_image_tuple):
      """Performs data augmentation on image.
      """
      label, image = label_image_tuple
      image = tf.pad(image, [[self._pad_size, self._pad_size],
                             [self._pad_size, self._pad_size], [0, 0]])
      image = tf.image.random_flip_left_right(image, seed=self._random_seed)
      image = tf.random_crop(image, [self._crop_size, self._crop_size, 3])
      return label, image

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      dataset = dataset.map(data_augmentation_fn)
    dataset = dataset.batch(self._batch_size)
    iterator = dataset.make_one_shot_iterator()
    labels, images = iterator.get_next()
    return {'labels': labels, 'images': images}

  def _get_per_pixel_mean(self, path):
    """Computes the per pixel mean (for each of the 3072) over all 50000
    training images.
    """
    filename_list = [os.path.join(path, fn) for fn in TRAIN_FILES]
    images = []
    for fn in filename_list:
      images.extend(self._read_file(fn))
    per_pixel_mean = np.array(images).astype(np.float32).mean(axis=0)
    return per_pixel_mean

  def _read_file(self, filename):
    """Reads a single cifar10 binary file and returns the list of raw byte
    sequences (rank-1 arrays with shape [32 * 32 * 3]).
    """
    content = np.fromfile(filename, np.uint8)
    images = []
    for i in np.arange(0, len(content), BYTES_PER_RECORD):
      images.append(content[i + 1 : i + BYTES_PER_RECORD])
    return images


class TrainerCifar10Dataset(Cifar10Dataset):
  """Cifar10 dataset for trainer.

  The dataset is mean-subtracted, shuffled, augmented, and batched.
  """
  def __init__(self,
               batch_size=128,
               pad_size=4,
               crop_size=32,
               buffer_size=10000,
               random_seed=0):
    """Constructor.

    Args:
      batch_size: int scalar, batch size.
      pad_size: int scalar, the num of pixels padded to both directions (low and 
        high) in the height and width dimension.
      crop_size: int scalar, the num of pixels to crop from the padded image in 
        the height and width dimension.
      buffer_size: int scalar, buffer size for shuffle operation. Must be large
        enough to get a sufficiently randomized sequence.
      random_seed: int scalar, random seed.
    """
    self._batch_size = batch_size
    self._pad_size = pad_size
    self._crop_size = crop_size
    self._buffer_size = buffer_size
    self._random_seed = random_seed

  @property
  def mode(self):
    return tf.contrib.learn.ModeKeys.TRAIN


class EvaluatorCifar10Dataset(Cifar10Dataset):
  """Cifar10 dataset for evaluator.

  The dataset is just mean-subtracted and batched. No shuffling or augmentation.
  """
  def __init__(self, batch_size=1):
    """Constructor.

    Args:
      batch_size: int scalar, batch size.
    """
    self._batch_size = batch_size

  @property
  def mode(self):
    return tf.contrib.learn.ModeKeys.EVAL

