# dcgan by liming @17.7.10

import tensorflow as tf
import os
import numpy as np
import math
import json
import simple_layers as lay

from src.liming_step1.train import Noise_ch, Noise_w, Noise_h

train_config_Noise_ch = Noise_ch
train_config_Noise_w = Noise_w
train_config_Noise_h = Noise_h

Fc_Channel = 64
De_Conv1_Channel = 32

Conv1_Channel = 32
Conv2_Chaneel = 64

train_fc_channel = Fc_Channel
train_De_Conv1_Channel = De_Conv1_Channel


def save_log(log_dir):
    json_log = {
        "Fc_Channel": Fc_Channel,
        "De_Conv1_Channel": De_Conv1_Channel,
        "Conv1_Channel": Conv1_Channel,
        "Conv2_Chaneel": Conv2_Chaneel,
    }
    with open(log_dir + '/model_config.json', 'w') as f:
        json.dump(json_log, f, indent=4)


def get_size(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def generate(z, h, w, ch, is_training, reuse, batch_size=200, train_config=None, model_config=None):
    if model_config is not None:
        De_Conv1_Channel = model_config['De_Conv1_Channel']
        Fc_Channel = model_config['Fc_Channel']
    else:
        Fc_Channel = train_fc_channel
        De_Conv1_Channel = train_De_Conv1_Channel

    if train_config is not None:
        Noise_h = train_config['Noise_h']
        Noise_w = train_config['Noise_w']
        Noise_ch = train_config['Noise_ch']
    else:
        Noise_h = train_config_Noise_h
        Noise_w = train_config_Noise_w
        Noise_ch = train_config_Noise_ch

    with tf.variable_scope('generator') as scope:
        h1, w1 = get_size(h, 2), get_size(w, 2)
        h2, w2 = get_size(h1, 2), get_size(w1, 2)
        # h3, w3 = get_size(h2, 1), get_size(w2, 1)
        # h4, w4 = get_size(h3, 2), get_size(w3, 2)
        z = tf.reshape(z, shape=[-1, Noise_ch * Noise_h * Noise_w], name='RESHAPE')
        fc0 = lay.fully_connect_layer(z, 'g_fc0_lin', h2 * w2 * Fc_Channel)
        fc0 = tf.reshape(fc0, [-1, h2, w2, Fc_Channel])
        fc0 = lay.batch_norm_official(fc0, is_training=is_training, reuse=reuse, name='g_bn0')
        fc0 = tf.nn.relu(fc0)

        # decon1 = lay.deconv_2d_layer(z, 'g_decon1', [5, 5, 32, 4], [batch_size, h3, w3, 32],
        #                              strides=[1, 2, 2, 1])
        # decon1 = lay.batch_norm_official(decon1, is_training=is_training, reuse=reuse, name='g_bn1')
        # decon1 = tf.nn.relu(decon1)
        #
        # decon2 = lay.deconv_2d_layer(decon1, 'g_decon2', [5, 5, 64, 32], [batch_size, h2, w2, 64],
        #                              strides=[1, 1, 1, 1])
        # decon2 = lay.batch_norm_official(decon2, is_training=is_training, reuse=reuse, name='g_bn2')
        # decon2 = tf.nn.relu(decon2)

        decon3 = lay.deconv_2d_layer(fc0, 'g_decon3', [5, 5, De_Conv1_Channel, Fc_Channel],
                                     [batch_size, h1, w1, De_Conv1_Channel],
                                     strides=[1, 2, 2, 1])
        decon3 = lay.batch_norm_official(decon3, is_training=is_training, reuse=reuse, name='g_bn3')
        decon3 = tf.nn.relu(decon3)

        decon4 = lay.deconv_2d_layer(decon3, 'g_decon4', [5, 5, ch, De_Conv1_Channel], [batch_size, h, w, ch],
                                     strides=[1, 2, 2, 1])
        decon4 = tf.nn.relu(decon4)
        return decon4


def decrim(x, is_training, reuse, batch_size=200):
    with tf.variable_scope('decriminator') as scope:
        conv0 = lay.conv_2d_layer(x, 'd_conv0', [5, 5, 16, Conv1_Channel], strides=[1, 2, 2, 1])
        conv0 = lay.batch_norm_official(conv0, is_training=is_training, reuse=reuse, name='d_bn0')
        conv0 = lay.leaky_relu(conv0)

        conv1 = lay.conv_2d_layer(conv0, 'd_conv1', [5, 5, Conv1_Channel, Conv2_Chaneel], strides=[1, 2, 2, 1])
        conv1 = lay.batch_norm_official(conv1, is_training=is_training, reuse=reuse, name='d_bn1')
        conv1 = lay.leaky_relu(conv1)

        # conv2 = lay.conv_2d_layer(conv1, 'd_conv2', [5, 5, 128, 256], strides=[1, 2, 2, 1])
        # conv2 = lay.batch_norm_official(conv2, is_training=is_training, reuse=reuse, name='d_bn2')
        # conv2 = lay.leaky_relu(conv2)
        #
        # conv3 = lay.conv_2d_layer(conv2, 'd_conv3', [5, 5, 256, 512], strides=[1, 2, 2, 1])
        # conv3 = lay.batch_norm_official(conv3, is_training=is_training, reuse=reuse, name='d_bn3')
        # conv3 = lay.leaky_relu(conv3)

        conv4_flatten = tf.reshape(conv1, [-1, Conv2_Chaneel * 2 * 2])

        fc4 = lay.fully_connect_layer(conv4_flatten, 'd_fc4', 1)

        return fc4
