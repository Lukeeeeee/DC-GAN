#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:38:33 2017

@author: lee
"""

import os
import scipy.misc
import numpy as np
from fmdcganpre import MDCGAN

from utils import pp, visualize, to_json, show_all_variables
import tensorflow as tf

with tf.Session() as sess:
    model = MDCGAN(sess, in_height = 1, in_width = 1, in_channel = 100,
                 out_height = 6, out_width = 6, out_channel = 512,
                 dlearning_rate = 0.001, glearning_rate = 0.001,
                 sample_num = 1000,
                 data_real_name = 'relu5_1', data_z_name = None,
                 epoch_num = 400, 
                 input_fname_pattern = '*.npy',
                 output_fname_pattern = '*.npy',  
                 batch_size = 64)
    model.train()