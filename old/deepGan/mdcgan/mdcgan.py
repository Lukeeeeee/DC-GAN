#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:45:33 2017

@author: lee
"""

from __future__ import division

from ops import *
from utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class mdcgan(object):
    def __init__(self, in_height, in_width, in_channel, out_height, out_width, out_channel, batch_size = 64):
        self.in_height = in_height
        self.in_width = in_width
        self.in_channel = in_channel
        self.out_height = out_height
        self.out_width = out_width
        self.out_channel = out_channel
        self.batch_size = batch_size
        self.build_model()
    
    def build_model(self):
        pass
    
    def discriminator(self, in_data):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(batch_norm(conv2d(in_data, self.channel, name="d_h0_conv")))
            h1 = lrelu(batch_norm(conv2d(h0, self.in_channel*2, name='d_h1_conv')))
            h2 = lrelu(batch_norm(conv2d(h1, self.in_channel*4, name='d_h2_conv')))
            h3 = lrelu(batch_norm(conv2d(h2, self.in_channel*8, name='d_h3_conv')))
            h4 = linear(batch_norm(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            return tf.nn.sigmoid(h4), h4
        '''
        build 64 128 256 512 deep conv with bias 
        
        with tf.name_scope("d_conv1") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, self.out_channel, 64], dtype = tf.float32, stddev = 0.1), name = "weights")
            conv = tf.nn.conv2d(in_data, kernel, [1,1,1,1], padding = "SAME")
            biases = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32), trainable = True, name = "biases")
            conv1 = tf.nn.relu(tf.nn.bias_add(conv + biases), name = scope)
        
        with tf.name_scope("d_conv2") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype = tf.float32, stddev = 0.1), name = "weights")
            conv = tf.nn.conv2d(conv1, kernel, [1,1,1,1], padding = "SAME")
            biases = tf.Variable(tf.constant(0.0, shape = [128], dtype = tf.float32), trainable = True, name = "biases")
            conv2 = tf.nn.relu(tf.nn.bias_add(conv + biases), name = scope)
        
        with tf.name_scope("d_conv3") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype = tf.float32, stddev = 0.1), name = "weights")
            conv = tf.nn.conv2d(conv2, kernel, [1,1,1,1], padding = "SAME")
            biases = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32), trainable = True, name = "biases")
            conv3 = tf.nn.relu(tf.nn.bias_add(conv + biases), name = scope)
            
        with tf.name_scope("d_conv4") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype = tf.float32, stddev = 0.1), name = "weights")
            conv = tf.nn.conv2d(conv3, kernel, [1,1,1,1], padding = "SAME")
            biases = tf.Variable(tf.constant(0.0, shape = [512], dtype = tf.float32), trainable = True, name = "biases")
            conv4 = tf.nn.relu(tf.nn.bias_add(conv + biases), name = scope)
        h4 = linear(tf.reshape(conv4, [self.batch_size, -1]), 1, 'd_h3_lin')
        return tf.nn.sigmoid(h4), h4
        '''
        
            
    def generate(self, in_data):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.out_height, self.out_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
            
            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))
            
            h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))
            
            h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)
    
    def sampler(self):
        pass
    
    def train(self):
        pass
    
    
        
        
    
    
    
    