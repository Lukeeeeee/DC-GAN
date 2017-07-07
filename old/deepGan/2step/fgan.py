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
import tensorflow as tf
from six.moves import xrange


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

in_height = 1
in_width = 1
in_channel = 100
out_height = 24
out_width = 24
out_channel = 256
dlearning_rate = 1e-3
glearning_rate = 0.00037
sample_num = 5000
data_real_name = 'relu3_1'
data_z_name = None
epoch_num = 400
input_fname_pattern = '*.npy'
output_fname_pattern = '*.npy'
batch_size = 64
'''change to real dataset path'''
data_real = glob(os.path.join("/media/geniuslee/My Passport/linux/vggtransfer/layers",
                              data_real_name,output_fname_pattern))[0:sample_num]
if data_z_name == None:
    pass
else:
    '''获取z的数据'''
    data_z = glob(os.path.join("./gdata", data_z_name, input_fname_pattern))

print(len(data_real))

inputs = tf.placeholder(tf.float32, [batch_size, out_height, out_width, out_channel],
                                 name='real_images')


# 是否在训练阶段
train_phase = tf.placeholder(tf.bool)
    #        sample_inputs = self.sample_inputs
if in_height == 1:
    z = tf.placeholder(tf.float32, [None, in_channel], name='z')
else:
    z = tf.placeholder(tf.float32, [None, in_height, in_width, in_channel], name='z')


def leaky_relu(x, alpha=0.1, name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(tf.multiply(alpha, x, name=name + 'lrelu/add'), x, name=name + 'lrelu/maxmium')


def batch_norm(x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):
    with tf.variable_scope(scope):
        # beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), trainable=True)
        # gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev), trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed

discriminator_variables_dict = {
    "W_1": tf.Variable(tf.truncated_normal([4, 4, 256, 512], stddev=0.002), name='Discriminator/W_1'),
    "b_1": tf.Variable(tf.constant(0.0, shape=[512]), name='Discriminator/b_1'),
    'beta_1': tf.Variable(tf.constant(0.0, shape=[512]), name='Discriminator/beta_1'),
    'gamma_1': tf.Variable(tf.random_normal(shape=[512], mean=1.0, stddev=0.02), name='Discriminator/gamma_1'),

    "W_2": tf.Variable(tf.truncated_normal([4, 4, 512, 512], stddev=0.002), name='Discriminator/W_2'),
    "b_2": tf.Variable(tf.constant(0.0, shape=[512]), name='Discriminator/b_2'),
    'beta_2': tf.Variable(tf.constant(0.0, shape=[512]), name='Discriminator/beta_2'),
    'gamma_2': tf.Variable(tf.random_normal(shape=[512], mean=1.0, stddev=0.02), name='Discriminator/gamma_2'),

    "W_3": tf.Variable(tf.truncated_normal([4, 4, 512, 1024], stddev=0.002), name='Discriminator/W_3'),
    "b_3": tf.Variable(tf.constant(0.0, shape=[1024]), name='Discriminator/b_3'),
    'beta_3': tf.Variable(tf.constant(0.0, shape=[1024]), name='Discriminator/beta_3'),
    'gamma_3': tf.Variable(tf.random_normal(shape=[1024], mean=1.0, stddev=0.02), name='Discriminator/gamma_3'),

    "W_4": tf.Variable(tf.truncated_normal([3*3*1024, 1], stddev=0.002), name='Discriminator/W_4'),
    "b_4": tf.Variable(tf.constant(0.0, shape=[1]), name='Discriminator/b_4'),
}


def discriminator(in_data):
    with tf.variable_scope("Discriminator"):
        out_1 = tf.nn.conv2d(in_data, discriminator_variables_dict['W_1'], strides=[1, 2, 2, 1],
                             padding='SAME')
        out_1 = tf.nn.bias_add(out_1, discriminator_variables_dict['b_1'])
        out_1 = batch_norm(out_1, discriminator_variables_dict['beta_1'], discriminator_variables_dict['gamma_1'],train_phase,
                           scope='bn_1')
        out_1 = leaky_relu(out_1, alpha=0.2, name="l_relu_1")

        out_2 = tf.nn.conv2d(out_1, discriminator_variables_dict['W_2'], strides=[1, 2, 2, 1],
                             padding='SAME')
        out_2 = tf.nn.bias_add(out_2, discriminator_variables_dict['b_2'])
        out_2 = batch_norm(out_2, discriminator_variables_dict['beta_2'], discriminator_variables_dict['gamma_2'],train_phase,
                           scope='bn_2')
        out_2 = leaky_relu(out_2, alpha=0.2, name="l_relu_2")

        out_3 = tf.nn.conv2d(out_2, discriminator_variables_dict['W_3'], strides=[1, 2, 2, 1],
                             padding='SAME')
        out_3 = tf.nn.bias_add(out_3, discriminator_variables_dict['b_3'])
        out_3 = batch_norm(out_3, discriminator_variables_dict['beta_3'], discriminator_variables_dict['gamma_3'],train_phase,
                           scope='bn_3')
        out_3 = leaky_relu(out_3, alpha=0.2, name="l_relu_3")

        re = tf.reshape(out_3, [-1, 3*3*1024])
        out_4 = tf.matmul(re, discriminator_variables_dict['W_4']) + discriminator_variables_dict['b_4']

        return tf.sigmoid(out_4)


'''生成'''

'''
def generator(z):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = out_height, out_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)

        # project `z` and reshape
        if (in_height == 1) & (in_width == 1):
            # z = tf.reshape(z, [-1])
            z_, h0_w, h0_b = ops.linear(z, 1024 * s_h2 * s_w2, 'g_h0_lin', with_w=True)
            h0 = tf.reshape(z_, [-1, s_h2, s_w2, 1024])
            h0 = ops.lrelu(g_bn0(h0))
            self.h1, self.h1_w, self.h1_b = ops.deconv2d(h0, [self.batch_size, self.out_height, self.out_width,
                                                              self.out_channel], name='g_h1', with_w=True)
        else:
            self.h1, self.h1_w, self.h1_b = ops.deconv2d(z, [self.batch_size, self.out_height, self.out_width,
                                                             self.out_channel], name='g_h1', with_w=True)
        return tf.nn.relu(self.g_bn1(self.h1)), tf.nn.tanh(self.h1)
'''

s_h1, s_w1 = out_height, out_width
s_h2, s_w2 = conv_out_size_same(s_h1, 2), conv_out_size_same(s_w1, 2)
s_h3, s_w3 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
s_h4, s_w4 = conv_out_size_same(s_h3, 2), conv_out_size_same(s_w3, 2)

print('4:',s_h4)
print('3:',s_w3)
print('2:',s_w2)
print('1:',s_w1)

generator_variables_dict = {
    "W_1": tf.Variable(tf.truncated_normal([in_channel, 1024*s_h4*s_w4], stddev=0.02), name='Generator/W_1'),
    "b_1": tf.Variable(tf.constant(0.0, shape=[1024*s_w4*s_h4]), name='Generator/b_1'),
    'beta_1': tf.Variable(tf.constant(0.0, shape=[1024]), name='Generator/beta_1'),
    'gamma_1': tf.Variable(tf.random_normal(shape=[1024], mean=1.0, stddev=0.02), name='Generator/gamma_1'),

    "W_2": tf.Variable(tf.truncated_normal([5, 5, 512, 1024], stddev=0.02), name='Generator/W_2'),
    "b_2": tf.Variable(tf.constant(0.0, shape=[512]), name='Generator/b_2'),
    'beta_2': tf.Variable(tf.constant(0.0, shape=[512]), name='Generator/beta_2'),
    'gamma_2': tf.Variable(tf.random_normal(shape=[512], mean=1.0, stddev=0.02), name='Generator/gamma_2'),

    "W_3": tf.Variable(tf.truncated_normal([5, 5, 512, 512], stddev=0.02), name='Generator/W_3'),
    "b_3": tf.Variable(tf.constant(0.0, shape=[512]), name='Generator/b_3'),
    'beta_3': tf.Variable(tf.constant(0.0, shape=[512]), name='Generator/beta_3'),
    'gamma_3': tf.Variable(tf.random_normal(shape=[512], mean=1.0, stddev=0.02), name='Generator/gamma_3'),

    "W_4": tf.Variable(tf.truncated_normal([5, 5, 256, 512], stddev=0.02), name='Generator/W_4'),
    "b_4": tf.Variable(tf.constant(0.0, shape=[256]), name='Generator/b_4'),
    'beta_4': tf.Variable(tf.constant(0.0, shape=[256]), name='Generator/beta_4'),
    'gamma_4': tf.Variable(tf.random_normal(shape=[256], mean=1.0, stddev=0.02), name='Generator/gamma_4'),

}


# Generator
def generator(z):
    with tf.variable_scope("Generator"):
        out_1 = tf.matmul(z, generator_variables_dict["W_1"]) + generator_variables_dict['b_1']
        out_1 = tf.reshape(out_1, [-1, s_h4, s_w4, 1024])
        out_1 = batch_norm(out_1, generator_variables_dict["beta_1"], \
                           generator_variables_dict["gamma_1"], train_phase,scope='bn_1')
        out_1 = tf.nn.relu(out_1, name='relu_1')



        out_2 = tf.nn.conv2d_transpose(out_1, generator_variables_dict['W_2'], \
                                       output_shape=tf.stack(
                                           [64, s_h3, s_w3, 512]),
                                       strides=[1, 2, 2, 1], padding='SAME')
        out_2 = tf.nn.bias_add(out_2, generator_variables_dict['b_2'])
        out_2 = batch_norm(out_2, generator_variables_dict["beta_2"], \
                           generator_variables_dict["gamma_2"],  train_phase,scope='bn_2')
        out_2 = tf.nn.relu(out_2, name='relu_2')



        out_3 = tf.nn.conv2d_transpose(out_2, generator_variables_dict['W_3'], \
                                       output_shape=tf.stack(
                                           [64, s_h2, s_w2, 512]),
                                       strides=[1, 2, 2, 1], padding='SAME')
        out_3 = tf.nn.bias_add(out_3, generator_variables_dict['b_3'])
        out_3 = batch_norm(out_3, generator_variables_dict["beta_3"], \
                           generator_variables_dict["gamma_3"],  train_phase,scope='bn_3')
        out_3 = tf.nn.relu(out_3, name='relu_3')



        out_4 = tf.nn.conv2d_transpose(out_3, generator_variables_dict['W_4'], \
                                       output_shape=tf.stack(
                                           [64, s_h1, s_w1, 256]),
                                       strides=[1, 2, 2, 1], padding='SAME')
        out_4 = tf.nn.bias_add(out_4, generator_variables_dict['b_4'])
        out_4 = batch_norm(out_4, generator_variables_dict["beta_4"], \
                           generator_variables_dict["gamma_4"],  train_phase,scope='bn_4')
        out_4 = tf.nn.relu(out_4, name='relu_4')

        return out_4


G = generator(z)
D = discriminator(inputs)
D_ = discriminator(G)


d_loss_real = tf.log(D )
d_loss_fake = tf.log(1 - D_ +0.0001)
g_loss = d_loss_fake
d_loss = -(d_loss_real + d_loss_fake + 0.0001)

# loss_train_G = (1 / 2) * (D_recog_fake - 1) ** 2
# loss_train_D = (1 / 2) * (D_recog_real - 1) ** 2 + (1 / 2) * (D_recog_fake) ** 2


d_optim = tf.train.AdamOptimizer(dlearning_rate).minimize(d_loss)
g_optim = tf.train.AdamOptimizer(glearning_rate).minimize(g_loss)

start_time = time.time()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer(), feed_dict={train_phase: True})
    for epoch in xrange(epoch_num):
        #            self.data_real = glob(os.path.join("/home/桌面/vggtransfer/layers/", dataset_name, self.input_fname_pattern))

        batch_idx = int(len(data_real) / batch_size)
        print(len(data_real))
        getshape = np.load(data_real[0])
        shape_height = getshape.shape[1]
        shape_width = getshape.shape[2]
        shape_channel = getshape.shape[3]

        if in_height == 1:
            pass
        else:
            getshape_z = np.load(data_z[0])
            shape_height_z = getshape_z.shape[0]
            shape_width_z = getshape_z.shape[1]
            shape_channel_z = getshape_z.shape[2]

        for idx in xrange(0, batch_idx):
            '''获取real数据分布'''
            batch_files = data_real[idx * batch_size: (idx + 1) * batch_size]
            batch_real = [np.load(batch_file).reshape(shape_height, shape_width, shape_channel) for batch_file in
                          batch_files]
            batch_inputs = np.array(batch_real).astype(np.float32)
            #print(batch_inputs)
            '''获取z数据分布'''
            if in_height == 1:
                batch_z = np.random.uniform(-1, 1, [batch_size, in_channel]).astype(np.float32)
                print(batch_z.shape)
            else:
                batch_z_files = data_z[idx * batch_size: (idx + 1) * batch_size]
                batch_z_temp = [np.load(batch_z_file).reshape(shape_height_z, shape_width_z, shape_channel_z) for
                                batch_z_file in batch_z_files]
                batch_z = np.array(batch_z_temp).astype(np.float32)
            '''更新一次 D network'''
            errD, _ =sess.run([d_loss, d_optim],
                          feed_dict={inputs: batch_inputs, z: batch_z,train_phase: True})
            # self.writer.add_summary(summary_str, counter)

            '''Update G network 更新两次'''
            errG, _ =sess.run([g_loss, g_optim],
                           feed_dict={inputs: batch_inputs,z: batch_z, train_phase: True})
            # # self.writer.add_summary(summary_str, counter)
            #
            #
            errG, _ = sess.run([g_loss, g_optim],
                           feed_dict={inputs: batch_inputs,z: batch_z, train_phase: True})
            # self.writer.add_summary(summary_str, counter)


            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (epoch, idx, batch_idx,
                     time.time() - start_time, np.mean(errD), np.mean(errG)))

            '''生成最后的数据并保存'''
            if epoch == 399:
                print('begin~~~~~~~')
                g_samples= sess.run( G, feed_dict={
                    z: batch_z,
                    inputs: batch_inputs,train_phase: True})

                print(g_samples)
                print(g_samples.shape)
                q = 0
                for gen in g_samples:
                    name = batch_files[q][:-4]
                    print(len(name))
                    print(gen.shape)
                    # np.save('./gdata/' + data_real_name + "/" + name[-30:] + ".npy", gen)
                    q += 1
                    print("Finally: " + name[-30:] + ".npy stored succeed!!!")

                print("all done  !!!!!!!")

    print("trained over!")


    