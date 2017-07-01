#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:30:18 2017

@author: lee
"""

from __future__ import division

import math
import os
import time
from glob import glob

import numpy as np
import scipy.misc as misc
import tensorflow as tf
from six.moves import xrange

import ops as ops
import utils as utils


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class MDCGAN(object):
    def __init__(self, sess, in_height, in_width, in_channel,
                 out_height, out_width, out_channel,
                 dlearning_rate, glearning_rate,
                 data_real_name, data_z_name = None,
                 sample_num = 1000,
                 epoch_num = 2, 
                 input_fname_pattern = '*.npy',
                 output_fname_pattern = '*.npy',  
                 batch_size = 64):
        self.in_height = in_height
        self.in_width = in_width
        self.in_channel = in_channel
        self.out_height = out_height
        self.out_width = out_width
        self.out_channel = out_channel
        self.batch_size = batch_size
        self.dlearning_rate = dlearning_rate
        self.glearning_rate = glearning_rate
        self.sample_num = sample_num
        self.epoch_num = epoch_num
        self.input_fname_pattern = input_fname_pattern
        self.output_fname_pattern = output_fname_pattern
        self.data_real_name = data_real_name
        self.data_z_name = data_z_name
        self.sess = sess
        
        self.g_bn0 = ops.batch_norm(name='g_bn0')
        self.g_bn1 = ops.batch_norm(name='g_bn1')
        self.g_bn2 = ops.batch_norm(name='g_bn2')
        self.g_bn3 = ops.batch_norm(name='g_bn3')
        
        
        self.d_bn0 = ops.batch_norm(name='d_bn0')
        self.d_bn1 = ops.batch_norm(name='d_bn1')
        self.d_bn2 = ops.batch_norm(name='d_bn2')
        self.d_bn3 = ops.batch_norm(name='d_bn3')
        self.d_bn4 = ops.batch_norm(name='d_bn4')
        self.d_bn5 = ops.batch_norm(name='d_bn5')
 #        self.checkpoint_dir = checkpoint_dir
        self.data_real = glob(os.path.join("/media/geniuslee/My Passport/linux/vggtransfer/resize/*"))[0:self.sample_num]
       
        if self.data_z_name == None:
            pass
        else:
            self.data_z = glob(os.path.join("./gdata", self.data_z_name, self.input_fname_pattern))
        #self.c_dim = imread(self.data[0]).shape[-1]
        self.build_model()
    
    def build_model(self):
        print(len(self.data_real))
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.out_height, self.out_width, self.out_channel], name='real_images')
#        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.out_height, self.out_width, self.out_channel], name='real_images')
#        self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num, self.out_height, self.out_width, self.out_channel], name='sample_inputs')
        inputs = self.inputs
#        sample_inputs = self.sample_inputs
        if self.in_height == 1:
            self.z = tf.placeholder(tf.float32, [None, self.in_channel], name = 'z')
        else:
            self.z = tf.placeholder(tf.float32, [None, self.in_height, self.in_width, self.in_channel], name='z')
        
        
        self.G, self.G_tanh = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(inputs)
#        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G_tanh, reuse=True)
        
        #self.d_sum = tf.summary.histogram("d", self.D)
        #self.d__sum = tf.summary.histogram("d_", self.D_)
        #self.G_sum = tf.summary.image("G", self.G)
        
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        
        #self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        #self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
                          
        #self.d_loss = self.d_loss_real + self.d_loss_fake

        #self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        #self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

    def train(self):
        d_optim = tf.train.AdamOptimizer(self.dlearning_rate).minimize(self.d_loss)
        g_optim = tf.train.AdamOptimizer(self.glearning_rate).minimize(self.g_loss)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        #self.g_sum = tf.summary.merge_all([self.d__sum,
        #                            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        #self.d_sum = tf.summary.merge_all([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
       # self.writer = tf.SummaryWriter("./logs", self.sess.graph)
        
        #counter = 1
        start_time = time.time()
            
   
        
        
        for epoch in xrange(self.epoch_num):
#            self.data_real = glob(os.path.join("/home/桌面/vggtransfer/layers/", dataset_name, self.input_fname_pattern))

            batch_idx = int(len(self.data_real)/self.batch_size)
            print(len(self.data_real))
 #           getshape = np.load(self.data_real[0])
  #          shape_height = getshape.shape[1]
   #         shape_width = getshape.shape[2]
    #        shape_channel = getshape.shape[3]
            
            if self.in_height == 1:
                pass
            else:
                getshape_z = np.load(self.data_z[0])
                shape_height_z = getshape_z.shape[0]
                shape_width_z = getshape_z.shape[1]
                shape_channel_z = getshape_z.shape[2]
                
            for idx in xrange(0,batch_idx): 
                
                batch_files = self.data_real[idx*self.batch_size : (idx+1)*self.batch_size]
                batch_real = [utils.get_image(batch_file,
                        input_height=192,
                        input_width=192,
                        resize_height=192,
                        resize_width=192) for batch_file in batch_files]
#                batch_real = [np.load(batch_file).reshape(shape_height, shape_width, shape_channel) for batch_file in batch_files]
                batch_inputs = np.array(batch_real).astype(np.float32)
                
                if self.in_height == 1:    
                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.in_channel]).astype(np.float32)
                    print (batch_z.shape)
                else:
                    batch_z_files = self.data_z[idx*self.batch_size : (idx+1)*self.batch_size]
                    batch_z_temp = [np.load(batch_z_file).reshape(shape_height_z, shape_width_z, shape_channel_z) for batch_z_file in batch_z_files]
                    batch_z = np.array(batch_z_temp).astype(np.float32)
                    
                self.sess.run([d_optim],
                              feed_dict={ self.inputs: batch_inputs, self.z: batch_z })
                #self.writer.add_summary(summary_str, counter)

                # Update G network
                self.sess.run([g_optim],
                              feed_dict={ self.z: batch_z })
                #self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                self.sess.run([g_optim],
                        feed_dict={ self.z: batch_z })
               # self.writer.add_summary(summary_str, counter)
          
                errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
                errD_real = self.d_loss_real.eval({ self.inputs: batch_inputs })
                errG = self.g_loss.eval({self.z: batch_z})
                
                
#                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idx,
                         time.time() - start_time, errD_fake+errD_real, errG))

                if epoch == 200:

                    #                    g_samples, g_samples_tanh = self.sess.run(self.generator, feed_dict = {
                    #                                                                         self.z : batch_z,
                    #                                                                        self.inputs : batch_inputs})
                    print('begin~~~~~~~')
                    g_samples, d_loss, g_loss = self.sess.run([self.G_tanh, self.d_loss, self.g_loss], feed_dict={
                        self.z: batch_z,
                        self.inputs: batch_inputs,
                    })
                    q = 0
                    for gen in g_samples:
                        name = batch_files[q][:-4]
                        print(len(name))
                        print(gen.shape)
                        gen +=1
                        gen *=127.5
                        gen = np.clip(gen,0,255).astype(np.uint8)
                        misc.imsave('./gdata/g200/' + name[-30:] + '.png', gen)

                        # np.save('./gdata/' + self.data_real_name + "/" + name[-30:] +".npy", gen)

                        q += 1
                        print("Finally: " + name[-30:] + ".npy stored succeed!!!")

                    print("one g_sample error  !!!!!!!")

                if epoch == 399:
                    
#                    g_samples, g_samples_tanh = self.sess.run(self.generator, feed_dict = {
 #                                                                         self.z : batch_z,
  #                                                                        self.inputs : batch_inputs})
                    print('begin~~~~~~~')
                    g_samples, d_loss, g_loss = self.sess.run( [self.G_tanh, self.d_loss, self.g_loss],feed_dict={
                                                                                    self.z: batch_z,
                                                                                    self.inputs: batch_inputs,                                                                                    
                                                                                    })
                    print('kasihile ')
                    print(g_samples)
                    print(g_samples.shape)
                    q = 0
                    for gen in g_samples:
                        name = batch_files[q][:-4]
                        print(len(name))
                        print(gen.shape)
                        gen += 1
                        gen *= 127.5
                        gen = np.clip(gen, 0, 255).astype(np.uint8)
                        misc.imsave('./gdata/g400/' + name[-30:] + '.png', gen)

    # np.save('./gdata/' + self.data_real_name + "/" + name[-30:] +".npy", gen)

                        q += 1
                        print("Finally: " + name[-30:] + ".npy stored succeed!!!")

                    
                    print("one g_sample error  !!!!!!!")



        print("trained over!")
            


    def discriminator(self, in_data, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
                
            if self.out_height == 6:
                h0 = ops.lrelu(self.d_bn0(ops.conv2d(in_data, 1024, name="d_h0_conv")))
                h1 = ops.linear(tf.reshape(h0, [self.batch_size, -1]), 1, 'd_h0_lin')
                return tf.nn.sigmoid(h1), h1
            
            if self.out_height == 12:
                h0 = ops.lrelu(self.d_bn0(ops.conv2d(in_data, 512, name="d_h0_conv")))
                h1 = ops.lrelu(self.d_bn1(ops.conv2d(h0, 1024, name='d_h1_conv')))
                h2 = ops.linear(tf.reshape(h1, [self.batch_size, -1]), 1, 'd_h2_lin')
                return tf.nn.sigmoid(h2), h2
            
            if self.out_height == 24:
                h0 = ops.lrelu(self.d_bn0(ops.conv2d(in_data, 512, name="d_h0_conv")))
                h1 = ops.lrelu(self.d_bn1(ops.conv2d(h0, 512, name='d_h1_conv')))
                h2 = ops.lrelu(self.d_bn2(ops.conv2d(h1, 1024, name='d_h2_conv')))
                h3 = ops.linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_h3_lin')
                return tf.nn.sigmoid(h3),h3
            
            if self.out_height == 48:
                h0 = ops.lrelu(self.d_bn0(ops.conv2d(in_data, 256, name="d_h0_conv")))
                h1 = ops.lrelu(self.d_bn1(ops.conv2d(h0, 512, name='d_h1_conv')))
                h2 = ops.lrelu(self.d_bn2(ops.conv2d(h1, 512, name='d_h2_conv')))
                h3 = ops.lrelu(self.d_bn3(ops.conv2d(h2, 1024, name='d_h3_conv')))
                h4 = ops.linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
                return tf.nn.sigmoid(h4),h4
            
            if self.out_height == 96:
                h0 = ops.lrelu(self.d_bn0(ops.conv2d(in_data, 128, name="d_h0_conv")))
                h1 = ops.lrelu(self.d_bn1(ops.conv2d(h0, 256, name='d_h1_conv')))
                h2 = ops.lrelu(self.d_bn2(ops.conv2d(h1, 512, name='d_h2_conv')))
                h3 = ops.lrelu(self.d_bn3(ops.conv2d(h2, 512, name='d_h3_conv')))
                h4 = ops.lrelu(self.d_bn4(ops.conv2d(h3, 1024, name='d_h4_conv')))
                h5 = ops.linear(tf.reshape(h4, [self.batch_size, -1]), 1, 'd_h4_lin')
                return tf.nn.sigmoid(h5), h5
            
            if self.out_height == 192:
                h0 = ops.lrelu(self.d_bn0(ops.conv2d(in_data, 128, name="d_h0_conv")))
                h1 = ops.lrelu(self.d_bn1(ops.conv2d(h0, 256, name='d_h1_conv')))
                h2 = ops.lrelu(self.d_bn2(ops.conv2d(h1, 512, name='d_h2_conv')))
                h3 = ops.lrelu(self.d_bn3(ops.conv2d(h2, 512, name='d_h3_conv')))
                h4 = ops.lrelu(self.d_bn4(ops.conv2d(h3, 1024, name='d_h4_conv')))
                h5 = ops.lrelu(self.d_bn5(ops.conv2d(h4, 1024, name='d_h5_conv')))
                h6 = ops.linear(tf.reshape(h5, [self.batch_size, -1]), 1, 'd_h5_lin')
                return tf.nn.sigmoid(h6), h6
                

            
    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.out_height, self.out_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
           
            
            # project `z` and reshape
            if (self.in_height == 1) & (self.in_width == 1):
                #z = tf.reshape(z, [-1])
                self.z_, self.h0_w, self.h0_b = ops.linear(z, 1024 * s_h2 * s_w2, 'g_h0_lin', with_w=True)
                self.h0 = tf.reshape(self.z_, [-1, s_h2, s_w2, 1024])
                h0 = ops.lrelu(self.g_bn0(self.h0))
                self.h1, self.h1_w, self.h1_b = ops.deconv2d(h0, [self.batch_size, self.out_height, self.out_width, self.out_channel], name='g_h1', with_w=True)
            else:
                self.h1, self.h1_w, self.h1_b = ops.deconv2d(z, [self.batch_size, self.out_height, self.out_width, self.out_channel], name='g_h1', with_w=True)
            return tf.nn.relu(self.g_bn1(self.h1)), tf.nn.tanh(self.h1)
        
        
        
    
    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            
            s_h, s_w = self.out_height, self.out_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
           
            
            # project `z` and reshape
            if (self.in_height == 1) & (self.in_width == 1):
                #z = tf.reshape(z, [-1])
                self.z_, self.h0_w, self.h0_b = ops.linear(z, 1024 * s_h2 * s_w2, 'g_h0_lin', with_w=True)
                self.h0 = tf.reshape(self.z_, [-1, s_h2, s_w2, 1024])
                h0 = ops.lrelu(self.g_bn0(self.h0))
                self.h1, self.h1_w, self.h1_b = ops.deconv2d(h0, [self.batch_size, self.out_height, self.out_width, self.out_channel], name='g_h1', with_w=True)
            else:
                self.h1, self.h1_w, self.h1_b = ops.deconv2d(z, [self.batch_size, self.out_height, self.out_width, self.out_channel], name='g_h1', with_w=True)
            return tf.nn.relu(self.g_bn1(self.h1)), tf.nn.tanh(self.h1)
        

