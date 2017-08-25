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
        # output 224 224 3
        h1, w1 = get_size(h, 2), get_size(w, 2)  # 112 112 64
        h2, w2 = get_size(h1, 2), get_size(w1, 2)  # 56 56 128
        h3, w3 = get_size(h2, 2), get_size(w2, 2)  # 28 28 256
        h4, w4 = get_size(h3, 2), get_size(w3, 2)  # 14 14 512
        h5, w5 = get_size(h4, 2), get_size(w4, 2)  # 7 7 512

        fc_ch = 512
        deconv1_out_ch = 512
        deconv2_out_ch = 256
        deconv3_out_ch = 128
        deconv4_out_ch = 64
        deconv5_out_ch = ch

        z = tf.reshape(z, shape=[-1, Noise_ch * Noise_h * Noise_w], name='RESHAPE')
        fc0 = lay.fully_connect_layer(z, 'g_fc0_lin', h5 * w5 * fc_ch)
        fc0 = tf.reshape(fc0, [-1, h5, w5, fc_ch])
        fc0 = lay.batch_norm_official(fc0, is_training=is_training, reuse=reuse, name='g_bn0')
        fc0 = tf.nn.relu(fc0)

        decon1 = lay.deconv_2d_layer(fc0, 'g_decon1', [5, 5, deconv1_out_ch, fc_ch],
                                     [batch_size, h4, w4, deconv1_out_ch],
                                     strides=[1, 2, 2, 1])
        decon1 = lay.batch_norm_official(decon1, is_training=is_training, reuse=reuse, name='g_bn1')
        decon1 = tf.nn.relu(decon1)

        decon2 = lay.deconv_2d_layer(decon1, 'g_decon2', [5, 5, deconv2_out_ch, deconv1_out_ch],
                                     [batch_size, h3, w3, deconv2_out_ch],
                                     strides=[1, 2, 2, 1])
        decon2 = lay.batch_norm_official(decon2, is_training=is_training, reuse=reuse, name='g_bn2')
        decon2 = tf.nn.relu(decon2)

        decon3 = lay.deconv_2d_layer(decon2, 'g_decon3', [5, 5, deconv3_out_ch, deconv2_out_ch],
                                     [batch_size, h2, w2, deconv3_out_ch],
                                     strides=[1, 2, 2, 1])
        decon3 = lay.batch_norm_official(decon3, is_training=is_training, reuse=reuse, name='g_bn3')
        decon3 = tf.nn.relu(decon3)

        decon4 = lay.deconv_2d_layer(decon3, 'g_decon', [5, 5, deconv4_out_ch, deconv3_out_ch],
                                     [batch_size, h1, w1, deconv4_out_ch],
                                     strides=[1, 2, 2, 1])
        decon4 = lay.batch_norm_official(decon4, is_training=is_training, reuse=reuse, name='g_bn4')
        decon4 = tf.nn.relu(decon4)

        decon5 = lay.deconv_2d_layer(decon4, 'g_decon4', [5, 5, deconv5_out_ch, deconv4_out_ch],
                                     [batch_size, h, w, deconv5_out_ch],
                                     strides=[1, 2, 2, 1])
        decon5 = tf.nn.tanh(decon5)
        return decon5, decon4, decon3, decon2, decon1


def decrim_list(input_list, is_training, reuse):
    res = []
    for input in input_list:
        size = input.get_shape().as_list()
        h = size[1]
        w = size[2]
        ch = size[3]
        out = decrim(input, w, h, ch, is_training=is_training, reuse=reuse, name=("%s_%s" % (str(h), str(w))))
        res.append(out)
    return res


def decrim(x, w, h, ch, is_training, reuse, name):

    with tf.variable_scope('decriminator') as scope:
        conv0 = lay.conv_2d_layer(x, name + 'd_conv0', [5, 5, ch, ch], strides=[1, 2, 2, 1])
        conv0 = lay.batch_norm_official(conv0, is_training=is_training, reuse=reuse, name=name + 'd_bn0')
        conv0 = lay.leaky_relu(conv0)

        conv1 = lay.conv_2d_layer(conv0, name + 'd_conv1', [5, 5, ch, ch], strides=[1, 2, 2, 1])
        conv1 = lay.batch_norm_official(conv1, is_training=is_training, reuse=reuse, name=name + 'd_bn1')
        conv1 = lay.leaky_relu(conv1)

        h1, w1 = get_size(h, 2), get_size(w, 2)
        h2, w2 = get_size(h1, 2), get_size(w1, 2)

        conv1_flatten = tf.reshape(conv1, [-1, w2 * h2 * ch])

        fc1 = lay.fully_connect_layer(conv1_flatten, name + 'd_fc1', 1)

        return fc1
