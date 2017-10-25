import os
import numpy as np

PATH = "/home/chaoji/Desktop/cifar10_data/cifar-10-batches-bin" 
HEIGHT = 32
WIDTH = 32
CHANNELS = 3
BYTES_PER_CHANNEL_IMG = HEIGHT * WIDTH 
BYTES_PER_IMG = BYTES_PER_CHANNEL_IMG * CHANNELS + 1
BATCH_SIZE = 200 
NUM_CLASSES = 10

class DataReader(object):
  def __init__(self, path, one_hot=True):
    self.path = path
    self.one_hot = one_hot

  def read_data(self):
    path = self.path
    filenames = [os.path.join(path, "data_batch_%d.bin" % i) for i in xrange(1, 6)]
    filenames += [os.path.join(path, "test_batch.bin")]
    tmp = []
    for fn in filenames[:-1]:
      images, labels = self._read_file(fn)
      tmp.append([images, labels])
    train = [np.vstack(zip(*tmp)[0]), np.hstack(zip(*tmp)[1])]
    test = self._read_file(filenames[-1])
    if self.one_hot:
      train[1], test[1] = self._to_one_hot(train[1]), self._to_one_hot(test[1])
    return train, test

  def _read_file(self, filename):
    batch = np.fromfile(filename, np.uint8)
    labels = []
    images = []
    for i in np.arange(0, len(batch), BYTES_PER_IMG):
      labels.append(batch[i])
      images.append(batch[i + 1 : i + BYTES_PER_IMG].reshape((CHANNELS, HEIGHT, WIDTH)))
    images = np.array(images).transpose(0, 2, 3, 1)
    labels = np.array(labels)
    return [images, labels]

  def _to_one_hot(self, labels):
    one_hot = np.zeros((len(labels), NUM_CLASSES))
    one_hot[np.arange(len(labels)), labels] = 1.
    return one_hot


class DataFeeder(object):
  def __init__(self, param, batch_size, proc_img, seed):   
    self.param = param
    self.seed = seed
    self.batch_size = batch_size
    self.proc_img = proc_img
    self._rs = np.random.RandomState(self.seed)

  def _random_crop(self, image, crop_size):
    h_crop, w_crop = crop_size
    h_img, w_img = image.shape[:2]
    h = self._rs.randint(0, h_img - h_crop + 1)
    w = self._rs.randint(0, w_img - w_crop + 1)
    return image[h:h+h_crop, w:w+w_crop]

  def _random_flip_left_right(self, image):
    if self._rs.binomial(1, .5) == 1:
      image = image[:, ::-1]
    return image

  def _random_brightness(self, image, max_delta):
    delta = self._rs.uniform(-max_delta, max_delta)
    image += delta
    return image

  def _random_contrast(self, image, lower, upper):
    contrast_factor = self._rs.uniform(lower, upper)
    old_shape = image.shape
    image = image.reshape((-1, image.shape[-1]))
    new_image = (image - image.mean(axis=0)) * contrast_factor + image.mean(axis=0)
    return new_image.reshape(old_shape)

  def _per_image_standardization(self, image):
    stddev = image.std()
    stddev = 1. if stddev == 0 else stddev
    adjusted_stddev = np.maximum(stddev, 1./np.sqrt(np.prod(image.shape)))
    return (image - image.mean()) / adjusted_stddev

  def _center_crop(self, image, crop_size):
    h_crop, w_crop = crop_size
    h_img, w_img = image.shape[:2]
    h = (h_img - h_crop) / 2
    w = (w_img - w_crop) / 2
    return image[h:h+h_crop, w:w+w_crop]

  def _pad_image(self, image, size):
    pad_height, pad_width = size
    pad_image = np.zeros((image.shape[0] + 2 * pad_height, 
                          image.shape[1] + 2 * pad_width, image.shape[2]))
    pad_image[pad_height:-pad_height, pad_width:-pad_width] = image
    return pad_image

  def preprocess_train_image(self, image):
    pad_size = self.param["pad_size"]
    crop_size = self.param["crop_size"]
    max_delta = self.param["max_delta"]
    lower = self.param["lower"]
    upper = self.param["upper"]
     
    image = image.astype(np.float32) 

    if self.proc_img["random_brightness"]:
      image = self._random_brightness(image, max_delta)
    if self.proc_img["random_contrast"]:
      image = self._random_contrast(image, lower, upper)
    if self.proc_img["per_image_standardization"]:
      image = self._per_image_standardization(image)
    if self.proc_img["pad_image"]:
      image = self._pad_image(image, pad_size)
    if self.proc_img["random_flip_left_right"]:
      image = self._random_flip_left_right(image)
    if self.proc_img["random_crop"]:
      image = self._random_crop(image, crop_size)

    return image[np.newaxis, :]

  def preprocess_test_image(self, image):
    crop_size = self.param["crop_size"]
    if self.proc_img["center_crop"]:
      image = self._center_crop(image, crop_size)
    if self.proc_img["per_image_standardization"]:
      image = self._per_image_standardization(image)
    return image[np.newaxis, :]

  def generate_train_batch(self, images, labels):
    indexes = self._rs.choice(images.shape[0], self.batch_size, False)
    batch = images[indexes]
    batch = np.vstack([self.preprocess_train_image(img) for img in batch])
    return batch, labels[indexes]
