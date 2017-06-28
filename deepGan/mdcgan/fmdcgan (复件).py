#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:30:18 2017

@author: lee
"""

from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from ops import *
from utils import *

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class MDCGAN(object):
    def __init__(self, sess, in_height, in_width, in_channel,
                 out_height, out_width, out_channel,
                 dlearning_rate, glearning_rate,
                 data_real_name, data_z_name = None,
                 epoch_num = 400, 
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
        self.epoch_num = epoch_num
        self.input_fname_pattern = input_fname_pattern
        self.data_real_name = data_real_name
        self.data_z_name = data_z_name
        self.checkpoint_dir = checkpoint_dir
        self.data_real = glob(os.path.join("/home/lee/桌面/vggtransfer/layers", self.data_real_name, self.output_fname_pattern))
        if self.data_z_name == None:
            pass
        else:
            self.data_z = glob(os.path.join("./gdata", self.data_z_name, self.input_fname_pattern))
        #self.c_dim = imread(self.data[0]).shape[-1]
        self.build_model()
    
    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.out_height, self.out_width, self.out_channel], name='real_images')
        self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num, self.out_height, self.out_width, self.out_channel], name='sample_inputs')
        inputs = self.inputs
        sample_inputs = self.sample_inputs
        self.z = tf.placeholder(tf.float32, [None, self.in_height, self.in_width, self.in_channel], name='z')

        self.G, self.G_tanh = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(inputs)
        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

    def train(self):
        d_optim = tf.train.AdamOptimizer(self.dlearning_rate).minimize(self.d_loss)
        g_optim = tf.train.AdamOptimizer(self.glearning_rate).minimize(self.g_loss)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        
        counter = 1
        start_time = time.time()
            
'''    
        if (self.in_height == 1) & (self.in_width == 1):
            sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.in_height, self.in_width,self.in_channel))
        else:
            pass
        
        getshape = np.load(self.data_real[0])
        shape_height = getshape.shape[1]
        shape_width = getshape.shape[2]
        shape_channel = getshape.shape[3]
        
        sample_files = self.data[0:self.sample_num]
        sample = [np.load(sample_file).reshape(shape_height, shape_width, shape_channel) for sample_file in sample_files]
        sample_inputs = np.array(sample).astype(np.float32)
'''
        
        
        
        for epoch in xrange(self.epoch_num):
#            self.data_real = glob(os.path.join("/home/桌面/vggtransfer/layers/", dataset_name, self.input_fname_pattern))
            batch_idx = int(len(self.data_real)/self.batch_size)
            
            getshape = np.load(self.data_real[0])
            shape_height = getshape.shape[1]
            shape_width = getshape.shape[2]
            shape_channel = getshape.shape[3]
            
            if self.in_height == 1:
                pass
            else:
                getshape_z = np.load(self.data_z)
                shape_height_z = getshape_z.shape[1]
                shape_width_z = getshape_z.shape[2]
                shape_channel_z = getshape_z.shape[3]
                
            for idx in xrange(0,batch_idx): 
                
                batch_files = self.data_real[idx*self.batch_size : (idx+1)*self.batch_size]
                batch_real = [np.load(batch_file).reshape(shape_height, shape_width, shape_channel) for batch_file in batch_files]
                batch_inputs = np.array(batch_real).astype(np.float32)
                
                if self.in_height == 1:    
                    batch_z = np.random.uniform(-1, 1,[self.batch_size, self.in_height, self.in_width, self.in_channel]).astype(np.float32)
                else:
                    batch_z_files = self.data_z[idx*self.batch_size : (idx+1)*self.batch_size]
                    batch_z_temp = [np.load(batch_z_file).reshape(shape_height_z, shape_width_z, shape_channel_z) for batch_z_file in batch_z_files]
                    batch_z = np.array(batch_z).astype(np.float32)
                    
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={ self.inputs: batch_images, self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)
          
                errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
                errD_real = self.d_loss_real.eval({ self.inputs: batch_inputs })
                errG = self.g_loss.eval({self.z: batch_z})
                
#                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake+errD_real, errG))
                
                if epoch == self.epoch_num-1:
                    try:
                        g_samples, d_loss, g_loss = self.sess.run([self.generate, self.d_loss,self.g_loss],
                                                                  feed_dict = {
                                                                          self.z : batch_z,
                                                                          self.inputs : batch_inputs})
                        for gen in g_samples:
                            q=0
                            name = batch_files[q][:-4]
                            np.save('./gdata/' + self.data_real_name + "/" + name[8:] +".npy", gen)
                            q++
                            print("Finally: "+ name[8:] + ".npy stored succeed!!!" )
                    except:
                        print("one g_sample error  !!!!!!!")
        print("trained over!")
            


    def discriminator(self, in_data):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
                
            if self.out_height == 6:
                h0 = lrelu(batch_norm(conv2d(in_data, 1024, name="d_h0_conv")))
                h1 = linear(batch_norm(h0, [self.batch_size, -1]), 1, 'd_h0_lin')
                return tf.nn.sigmoid(h1), h1
            
            if self.out_height == 12:
                h0 = lrelu(batch_norm(conv2d(in_data, 512, name="d_h0_conv")))
                h1 = lrelu(batch_norm(conv2d(h0, 1024, name='d_h1_conv')))
                h2 = linear(batch_norm(h1, [self.batch_size, -1]), 1, 'd_h2_lin')
                return tf.nn.sigmoid(h2), h2
            
            if self.out_height == 24:
                h0 = lrelu(batch_norm(conv2d(in_data, 512, name="d_h0_conv")))
                h1 = lrelu(batch_norm(conv2d(h0, 512, name='d_h1_conv')))
                h2 = lrelu(batch_norm(conv2d(h1, 1024, name='d_h2_conv')))
                h3 = linear(batch_norm(h2, [self.batch_size, -1]), 1, 'd_h3_lin')
                return tf.nn.sigmoid(h3),h3
            
            if self.out_height == 48:
                h0 = lrelu(batch_norm(conv2d(in_data, 256, name="d_h0_conv")))
                h1 = lrelu(batch_norm(conv2d(h0, 512, name='d_h1_conv')))
                h2 = lrelu(batch_norm(conv2d(h1, 512, name='d_h2_conv')))
                h3 = lrelu(batch_norm(conv2d(h2, 1024, name='d_h3_conv')))
                h4 = linear(batch_norm(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
                return tf.nn.sigmoid(h4),h4
            
            if self.out_height == 96:
                h0 = lrelu(batch_norm(conv2d(in_data, 128, name="d_h0_conv")))
                h1 = lrelu(batch_norm(conv2d(h0, 256, name='d_h1_conv')))
                h2 = lrelu(batch_norm(conv2d(h1, 512, name='d_h2_conv')))
                h3 = lrelu(batch_norm(conv2d(h2, 512, name='d_h3_conv')))
                h4 = lrelu(batch_norm(conv2d(h3, 1024, name='d_h4_conv')))
                h5 = linear(batch_norm(h4, [self.batch_size, -1]), 1, 'd_h4_lin')
                return tf.nn.sigmoid(h5), h5
            if self.out_height == 192:
                h0 = lrelu(batch_norm(conv2d(in_data, 128, name="d_h0_conv")))
                h1 = lrelu(batch_norm(conv2d(h0, 256, name='d_h1_conv')))
                h2 = lrelu(batch_norm(conv2d(h1, 512, name='d_h2_conv')))
                h3 = lrelu(batch_norm(conv2d(h2, 512, name='d_h3_conv')))
                h4 = lrelu(batch_norm(conv2d(h3, 1024, name='d_h4_conv')))
                h5 = lrelu(batch_norm(conv2d(h4, 1024, name='d_h5_conv')))
                h6 = linear(batch_norm(h5, [self.batch_size, -1]), 1, 'd_h5_lin')
                return tf.nn.sigmoid(h6), h6
                
                
                '''
                h0 = lrelu(batch_norm(conv2d(in_data, self.channel, name="d_h0_conv")))
                h1 = lrelu(batch_norm(conv2d(h0, self.in_channel*2, name='d_h1_conv')))
                h2 = lrelu(batch_norm(conv2d(h1, self.in_channel*4, name='d_h2_conv')))
                h3 = lrelu(batch_norm(conv2d(h2, self.in_channel*8, name='d_h3_conv')))
                h4 = linear(batch_norm(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
                return tf.nn.sigmoid(h4), h4
                '''    





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
        
            
    def generate(self, z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.out_height, self.out_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
           
            
            # project `z` and reshape
            if (self.in_height == 1) & (self.in_width == 1):
                self.z_, self.h0_w, self.h0_b = linear(z, 1024*s_h2*s_w2, 'g_h0_lin', with_w=True)
                self.h0 = tf.reshape(self.z_, [-1, s_h2, s_w2, self.out_channel*2])
                h0 = tf.nn.relu(batch_norm(self.h0))
                self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, self.out_height, self.out_width, self.out_channel], name='g_h1', with_w=True)
            else:
                self.h1, self.h1_w, self.h1_b = deconv2d(z, [self.batch_size, self.out_height, self.out_width, self.out_channel], name='g_h1', with_w=True)
            return tf.nn.relu(batch_norm(self.h1)), tf.nn.tanh(self.h1)
        
        
        
    
    def sampler(self):
        pass
 '''   
    def model_dir(self):
        return "{}_{}_{}_{}".format(self.dataset_name, self.batch_size, self.output_height, self.output_width)
      
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
        return False, 0

'''

