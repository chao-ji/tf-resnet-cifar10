import tensorflow as tf
import numpy as np
import os

TRAIN_FILES = ('data_batch_1.bin',
               'data_batch_2.bin',
               'data_batch_3.bin',
               'data_batch_4.bin',
               'data_batch_5.bin')
TEST_FILES = 'test_batch.bin'


class Cifar10DatasetBuilder(object):
  """Builds a tf.data.Dataset instance that batch the (optionally 
  data-augmentated) images. 
  """
  def __init__(self, 
               pad_size=4, 
               crop_size=32, 
               buffer_size=50000,
               random_seed=0):
    """Constrcutor.

    Args:
      pad_size: int scalar, the num of pixels padded to both directions (low and
        high) in the height and width dimension.
      crop_size: int scalar, the num of pixels to crop from the padded image in 
        the height and width dimension.
      buffer_size: int scalar, buffer size for shuffle operation. Must be large
        enough to get a sufficiently randomized sequence.
      random_seed: int scalar, random seed.
    """
    self._pad_size = pad_size
    self._crop_size = crop_size
    self._buffer_size = buffer_size
    self._random_seed = random_seed

  def build_dataset(self, labels, images, batch_size, training=True):
    """Builds the CIFAR10 dataset.

    Args:
      labels: numpy array of shape [num_images], holding the class labels/
      images: numpy array of shape [num_images, 32, 32, 3], holding the images.
      batch_size: int scalar, batch size.
      training: bool scalar, whether to build dataset for training (True) or 
        evaluation (False).

    Returns:
      dataset: a tf.data.Dataset instance.
    """
    dataset = tf.data.Dataset.from_tensor_slices((labels, images))

    if training:
      dataset = dataset.repeat().shuffle(
          buffer_size=self._buffer_size, seed=self._random_seed) 

    def data_augmentation_fn(*label_image_tuple):
      label, image = label_image_tuple
      image = tf.pad(image, [[self._pad_size, self._pad_size], 
                             [self._pad_size, self._pad_size], [0, 0]])
      image = tf.image.random_flip_left_right(image, seed=self._random_seed)
      image = tf.image.random_crop(image, [self._crop_size, self._crop_size, 3])
      return label, image

    if training:
      dataset = dataset.map(data_augmentation_fn)

    dataset = dataset.batch(batch_size)

    return dataset

def parse_binary(filename):
  """Parse CIFAR10 data in binary format.

  Args:
    filename: string scalar, the filename of a CIFAR10 data batch file in binary
      format.

  Returns:
    labels: numpy array of shape [num_images], holding the class labels/
    images: numpy array of shape [num_images, 32, 32, 3], holding the images.
  """
  content = np.fromfile(filename, 'uint8')
  labels, images = np.split(
      content.reshape(10000, -1), axis=1, indices_or_sections=[1])
  labels = np.squeeze(labels)
  images = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
  return labels, images

def read_data(data_path, training=True):
  """Reads CIFAR10 data and performs mean subtraction.

  Args:
    data_path: string scalar, the path to the directory holding CIFAR10 data 
      batch files.
    training: bool scalar, whether to read the training split (True) or 
      evaluation split (False).

  Returns:
    labels: numpy array of shape [num_images], holding the class labels/
    images: numpy array of shape [num_images, 32, 32, 3], holding the images.
  """
  filename_list = [os.path.join(data_path, fn) for fn in TRAIN_FILES]
  labels, images = tuple(zip(*[parse_binary(filename_list[i]) for i in range(5)]))
  labels = np.concatenate(labels, axis=0)
  images = np.concatenate(images, axis=0)

  per_pixel_mean = images.mean(axis=0)
  
  if not training:
    labels, images = parse_binary(os.path.join(data_path, TEST_FILES)) 
  images = images - per_pixel_mean

  labels = labels.astype('int64')
  images = images.astype('float32')

  return labels, images
