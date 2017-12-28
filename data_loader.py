import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os,errno
import math
import random
import shutil
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.framework import dtypes

class DataLoader(object):

    def __init__(self, directory, image_size=300, verbose=True):
        """
        Constructor initializes the directory and seeds random and np.random
        """
        self.directory = directory
        np.random.seed = 10
        random.seed = 100
        tf.set_random_seed(100)
        self.verbose = verbose
        self.df_y = None
        self.num_samples = 0
        self.num_train = 0
        self.num_test = 0
        self.y_train_df = None
        self.y_test_df = None
        self.train_data = None
        self.test_data = None
        self.NUM_CLASSES = 2
        self.IMAGE_SIZE = image_size
        self.files = None


    def read_data(self):
        self.df_y = pd.read_csv(self.directory+"/selfie_dataset.txt", sep="\s+",header=None,usecols=[0,3],names=['name','sex'])
        if self.verbose:
            print(self.df_y.shape)


    def create_train_test(self):
        """
        Create train and test samples from the original data set
        """
        self.num_samples = self.df_y.shape[0]
        self.num_train = math.floor(self.num_samples*0.9)
        self.num_test = self.num_samples - self.num_train
        if self.verbose:
            print("Total samples", self.num_samples)
            print("Total training samples", self.num_train)
            print("Total test samples", self.num_test)


    def shuffle(self):
        """
        Shuffle the contents of the directory
        """
        print("inside shuffle")
        self.files = [i for i in os.listdir(self.directory+'/images/')]
        if self.verbose:
            print(self.files[0])
            print("Length of files",len(self.files))
        self.files.remove(self.files[0])
        np.random.shuffle(self.files)
        try:
            os.mkdir(self.directory+'/images/Train')
            os.mkdir(self.directory+'/images/Test')
            for i,j in enumerate(self.files):
                if i <= self.num_train:
                    shutil.move(os.path.join(self.directory+'/images/',j), os.path.join(self.directory+'/images/Train/',j))
                else:
                    shutil.move(os.path.join(self.directory+'/images/',j), os.path.join(self.directory+'/images/Test/',j))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


    def create_train_test_df(self):
        """
        Create train and test df from the existing image directory
        """
        train_name = [i[:-4] for i in os.listdir(self.directory+'/images/Train/')]
        test_name = [i[:-4] for i in os.listdir(self.directory+'/images/Test/')]
        self.y_train_df = self.df_y[self.df_y['name'].isin(train_name)]
        self.y_test_df = self.df_y[self.df_y['name'].isin(test_name)]
        if self.verbose:
            print("y_train_shape", self.y_train_df.shape)
            print("y_test_shape", self.y_test_df.shape)
            print("100th train file name", self.y_train_df.iloc[100])
            print("100th test file name", self.y_test_df.iloc[100])


    def create_train_test_data(self):
        """
        Create train and test data from the data set
        """
        train_images = [os.path.abspath(os.path.join(os.sep,self.directory+'/images/Train/',i)) for i in os.listdir(self.directory+'/images/Train/')]
        test_images = [os.path.abspath(os.path.join(os.sep,self.directory+'/images/Test/',i)) for i in os.listdir(self.directory+'/images/Test/')]
        train_images_tf = convert_to_tensor(train_images,dtype=tf.string)
        test_images_tf = convert_to_tensor(test_images,dtype=tf.string)
        train_labels_tf = convert_to_tensor(list(self.y_train_df['sex']),dtype=dtypes.int32)
        test_labels_tf = convert_to_tensor(list(self.y_test_df['sex']),dtype = dtypes.int32)
        self.train_data = tf.data.Dataset.from_tensor_slices((train_images_tf,train_labels_tf))
        self.test_data  = tf.data.Dataset.from_tensor_slices((test_images_tf,test_labels_tf))


    def distort_img_train(self, img_file, labels):
        one_hot_labels = tf.one_hot(labels, self.NUM_CLASSES)
        img_str = tf.read_file(img_file)
        img_decode = tf.image.decode_jpeg(img_str, channels=3)
        img_resize = tf.image.resize_image_with_crop_or_pad(img_decode,target_height=self.IMAGE_SIZE,target_width=self.IMAGE_SIZE)
        distort_img = tf.image.random_flip_left_right(img_resize,seed=tf.set_random_seed(10))
        img = tf.image.random_brightness(distort_img,max_delta=2.0,seed=10)
        img_std = tf.image.per_image_standardization(img)

        return img_std,one_hot_labels


    def distort_img_test(self, img_file, labels):
        one_hot_labels = tf.one_hot(labels, self.NUM_CLASSES)
        img_str = tf.read_file(img_file)
        img_decode = tf.image.decode_jpeg(img_str,channels=3)
        img_resize = tf.image.resize_image_with_crop_or_pad(img_decode,target_height=self.IMAGE_SIZE,target_width=self.IMAGE_SIZE)
        img_std = tf.image.per_image_standardization(img_resize)

        return img_std,one_hot_labels


    def get_data(self, batch_size,is_train_data = True,num_threads = 6):
        if is_train_data:
            data = self.train_data.map(self.distort_img_train, num_parallel_calls = num_threads)
        else:
            data = self.test_data.map(self.distort_img_test, num_parallel_calls = num_threads)
        data = data.batch(batch_size)
        return data
