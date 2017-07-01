#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 17:37:38 2017

@author: lee
"""
from __future__ import division

import os
from glob import glob

import numpy as np
import tensorflow as tf
from six.moves import xrange

import utils as utils
import vgg as vgg

#parameters we should define
matpath = "imagenet-vgg-verydeep-19.mat"
pooling = "max"
image_path = "./faces"
input_fname_pattern = ".jpg"
batch_size = 64

#

Layers = ('conv1_1', 'relu1_1',
    'conv2_1', 'relu2_1',
    'conv3_1', 'relu3_1',
    'conv4_1', 'relu4_1',
    'conv5_1', 'relu5_1')

# Layers = ('relu1_1','relu2_1')

for i in Layers:
    print (i)
k = utils.imread('./faces/cf59317fe6cc4da9f6212eaaeb7416b7-0.jpg')
print(k.shape)

data = glob(os.path.join(image_path, '*.jpg'))
vgg_weights, vgg_mean_pixel = vgg.load_net(matpath)

# image = tf.placeholder(tf.float32, [None, 96, 96, 3])

data_num = int(len(data))
image_data0 = utils.imread(data[0]).astype(np.float32)
shape = (1,) + image_data0.shape
image = tf.placeholder(tf.float32, shape = shape)
net = vgg.net_preloaded(vgg_weights, image, pooling)

#    batch_files = dataset[idx*batch_size:(idx+1)*batch_size]
#    batch_data = [utils.imread(batchfile).astype(np.float32)
#               for batchfile in batch_files]
#    print (batch_data)


with tf.Session() as sess:
    for idx in xrange(0, 30000):
        '''读取图片'''
        image_data = utils.imread(data[idx]).astype(np.float32)
        shape = (1,) + image_data.shape
        name = data[idx][:-4]
#    image_pre = np.array([vgg.preprocess(batch_data, vgg_mean_pixel)]).astype(np.float32)
#    image_pre = image_pre.reshape([1, 96, 96, 3])
#    net = vgg.net_preloaded(vgg_weights, image_pre, pooling)
    
#    for layer in Layers:
        #        name = dataset[idx][:-4]
        features={}
        '''预处理，这个地方存疑'''
        image_pre = np.array([vgg.preprocess(image_data, vgg_mean_pixel)])
        for layer in Layers:
            features[layer] = net[layer].eval(feed_dict={image: image_pre})
            print(features[layer])
            np.save('./layers/' + layer + "/" + name[8:] + ".npy", features[layer])
            print("file:" + str(idx) + ", " + layer + "succeed stored")

#    for layer in Layers:
#        features[layer] = net[layer].eval(feed_dict={image: content_pre}

print("finished")
