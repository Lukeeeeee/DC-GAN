#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 20:14:15 2017

@author: lee
"""

import numpy as np
import tensorflow as tf
import glob as glob
import os


Layers = ('conv1_1', 'relu1_1',
    'conv2_1', 'relu2_1',
    'conv3_1', 'relu3_1',
    'conv4_1', 'relu4_1',
    'conv5_1', 'relu5_1')
np_path = './layers/conv1_1/'
#data = glob(os.path.join(np_path, '*.npy'))
for layer in Layers:
    conv1_1 = np.load('./layers/'+layer+'/0a4cbc873f873864d1a3093a3c39a544-0.npy')
    print(conv1_1.shape)