import numpy as np
from scipy import misc
import glob

class Cifar:
  def __init__(self, num_train_images, num_test_images):
    self.train_dir  = 'train'
    self.test_dir = 'test'
    self.labels_file = 'labels.txt'
    self.num_train_images  = num_train_images 
    self.num_test_images = num_test_images 
    self.width = 32 
    self.height = 32 
    self.channels = 3

  def get_label(self, filename):
    return self.label_map[filename.split("_")[1].split(".")[0]]

  def load_image_and_labels(self, dirname, num_images):
    files = glob.glob(dirname + '/*.png')[0 : num_images]
    images = np.zeros((num_images, self.width, self.height, self.channels), \
        dtype = np.uint8)
    for i in range(0, num_images):
      np.copyto(images[i], misc.imread(files[i]))
    image_labels = [self.get_label(filename) for filename in files]
    return (files, images, image_labels)

      
  def load_data(self):
    with open(self.labels_file) as f:
        lines = f.readlines()
    self.labels = [l.rstrip() for l in lines]
    self.label_map = { key : value for (key, value) in \
        zip(self.labels , range(0, len(self.labels))) }
    (self.train_files, self.train_images, self.train_labels) = \
        self.load_image_and_labels(self.train_dir, self.num_train_images)
    (self.test_files, self.test_images, self.test_labels) = \
        self.load_image_and_labels(self.test_dir, self.num_test_images)
    return ((self.train_images, self.train_labels), \
            (self.test_images, self.test_labels))



