import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class ResNet(object):

    def __init__(self, verbose=True):
        self.verbose = verbose
        tf.set_random_seed(0)


    def identity_block(self, X,f,filters,stage,block):
        conv_name_base = 'res'+str(stage)+block
        bn_name_base = 'bn'+str(stage)+block

        F1,F2,F3 = filters

        shortcut_input = X

        X = tf.layers.conv2d(X,filters=F1,kernel_size=(1,1),strides=(1,1),padding='valid',name=conv_name_base+'2a'
                             ,kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0))
        X = tf.layers.batch_normalization(X,axis=3,name=bn_name_base+'2a')

        X = tf.nn.relu(X)

        X = tf.layers.conv2d(X,filters=F2,kernel_size=(f,f),strides=(1,1),padding='same',name=conv_name_base+'2b'
                            ,kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0))

        X = tf.layers.batch_normalization(X,axis=3,name=bn_name_base+'2b')

        X = tf.nn.relu(X)

        X = tf.layers.conv2d(X,filters=F3,kernel_size=(1,1),strides=(1,1),padding='valid',name=conv_name_base+'2c'
                            ,kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0))

        X = tf.layers.batch_normalization(X,axis=3,name=bn_name_base+'2c')

        add_block = tf.add(shortcut_input,X)

        X = tf.nn.relu(add_block)

        return X


    def convolution_block(self, X,f,filters,s,stage,block):
        conv_name_base = 'res'+str(stage)+block
        bn_name_base = 'bn'+str(stage)+block

        F1,F2,F3 = filters

        short_input = X

        X= tf.layers.conv2d(X,filters=F1,kernel_size=(1,1),strides=(s,s),padding='valid',name=conv_name_base+'2a'
                            ,kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0))
        X = tf.layers.batch_normalization(X,axis=3,name=bn_name_base+'2a')
        X = tf.nn.relu(X)

        X= tf.layers.conv2d(X,filters=F1,kernel_size=(f,f),strides=(1,1),padding='same',name=conv_name_base+'2b'
                           ,kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0))
        X = tf.layers.batch_normalization(X,axis=3,name=bn_name_base+'2b')
        X = tf.nn.relu(X)

        X= tf.layers.conv2d(X,filters=F1,kernel_size=(1,1),strides=(1,1),padding='valid',name=conv_name_base+'2c'
                           ,kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0))
        X = tf.layers.batch_normalization(X,axis=3,name=bn_name_base+'2c')

        shortcut_X = tf.layers.conv2d(short_input,filters=F3,kernel_size=(1,1),strides=(s,s),padding='valid',name=conv_name_base+'1'
                                     ,kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0))
        shortcut_X = tf.layers.batch_normalization(shortcut_X,axis=3,name=bn_name_base+'1')

        add_block = tf.add(shortcut_X,X)

        X = tf.nn.relu(add_block)

        return X



    def resNet_50(self, X):

        ##implement zero padding here

        #stage_1
        X = tf.layers.conv2d(X,filters=64,kernel_size=(7,7),strides=(2,2),name='conv1')
        X = tf.layers.batch_normalization(X,axis=3,name='batch_norm1')
        X = tf.nn.relu(X)
        X = tf.layers.max_pooling2d(X,strides=(2,2),pool_size=(3,3))

        #stage_2
        X = convolution_block(X,f=3,s=1,filters=[64,64,256],stage=2,block='a')
        X = identity_block(X,filters=[64,64,256],stage=2,f=3,block='b')
        X = identity_block(X,filters=[64,64,256],stage=2,f=3,block='c')

        #stage_3
        X = convolution_block(X,f=3,s=2,filters=[128,128,512],stage=3,block='a')
        X = identity_block(X,filters=[128,128,512],stage=3,f=3,block='b')
        X = identity_block(X,filters=[128,128,512],stage=3,f=3,block='c')
        X = identity_block(X,filters=[128,128,512],stage=3,f=3,block='d')

        #stage_4
        X = convolution_block(X,f=3,s=2,filters=[256,256,1024],stage=4,block='a')
        X = identity_block(X,filters=[256,256,1024],stage=4,f=3,block='b')
        X = identity_block(X,filters=[256,256,1024],stage=4,f=3,block='c')
        X = identity_block(X,filters=[256,256,1024],stage=4,f=3,block='d')
        X = identity_block(X,filters=[256,256,1024],stage=4,f=3,block='e')
        X = identity_block(X,filters=[256,256,1024],stage=4,f=3,block='f')

        #stage_5
        X = convolution_block(X,f=3,s=2,filters=[512,512,2048],stage=5,block='a')
        X = identity_block(X,filters=[512,512,2048],stage=5,f=3,block='b')
        X = identity_block(X,filters=[512,512,2048],stage=5,f=3,block='c')

        X = tf.layers.average_pooling2d(X,pool_size=(2,2),strides=(1,1),name='avg_pool')
        X = tf.layers.flatten(X,name='flatten')
        X = tf.layers.dense(X,units=2,activation=tf.nn.softmax(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))

        print("X ",X)
        return X
