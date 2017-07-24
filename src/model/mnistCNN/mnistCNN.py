# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

import numpy as np
import tensorflow as tf

from log import LOG_PATH
from src.model.model import Model


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class MnistCNN(Model):
    def __init__(self, sess, config, data):
        super(MnistCNN, self).__init__(sess=sess, config=config, data=data)
        self.config = config

        ti = datetime.datetime.now()
        self.log_dir = (
        LOG_PATH + '/mnist/' + str(ti.month) + '-' + str(ti.day) + '-' + str(ti.hour) + '-' + str(ti.minute)
        + '-' + str(ti.second) + '/')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.model_dir = self.log_dir + 'model/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.input = tf.placeholder(tf.float32,
                                    [None, self.config.IN_WIDTH, self.config.IN_HEIGHT, self.config.IN_CHANNEL])
        self.label = tf.placeholder(tf.int32,
                                    [None])

        self.is_training = tf.placeholder(tf.bool)

        self.out, self.keep_prob, self.conv1, self.conv2, self.conv3 = self.create_model()

        self.loss, self.accuracy, self.optimize_loss, self.predication = self.create_training_method()

        self.model_saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

    def create_model(self):

        W_conv1 = weight_variable([self.config.FILTER_SIZE,
                                   self.config.FILTER_SIZE,
                                   self.config.IN_CHANNEL,
                                   self.config.CONV_LAYER_1_OUT_CHANNEL])
        b_conv1 = bias_variable([self.config.CONV_LAYER_1_OUT_CHANNEL])
        h_conv1 = tf.nn.relu(conv2d(self.input, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([self.config.FILTER_SIZE,
                                   self.config.FILTER_SIZE,
                                   self.config.CONV_LAYER_1_OUT_CHANNEL,
                                   self.config.CONV_LAYER_2_OUT_CHANNEL])
        b_conv2 = bias_variable([self.config.CONV_LAYER_2_OUT_CHANNEL])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_conv3 = weight_variable([self.config.FILTER_SIZE,
                                   self.config.FILTER_SIZE,
                                   self.config.CONV_LAYER_2_OUT_CHANNEL,
                                   self.config.CONV_LAYER_3_OUT_CHANNEL])
        b_conv3 = bias_variable([self.config.CONV_LAYER_3_OUT_CHANNEL])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        W_fc1 = weight_variable([self.config.FULLY_CONNECTED_IN, self.config.FULLY_CONNECTED_OUT])
        b_fc1 = bias_variable([self.config.FULLY_CONNECTED_OUT])

        h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool3, [-1, self.config.FULLY_CONNECTED_IN]), W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        if self.is_training is True:
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        else:
            h_fc1_drop = h_fc1

        # Map the 1024 features to 10 classes, one for each digit
        W_fc2 = weight_variable([self.config.FULLY_CONNECTED_OUT, self.config.OUTPUT_SIZE])
        b_fc2 = bias_variable([self.config.OUTPUT_SIZE])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv, keep_prob, h_pool1, h_pool2, h_pool3

    def create_training_method(self):
        label = tf.one_hot(indices=self.label, depth=self.config.OUTPUT_SIZE)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=self.out))
        optimize = tf.train.AdamOptimizer(learning_rate=self.config.LEARNING_RATE)
        optimize_loss = optimize.minimize(loss=loss)

        correct_predication = tf.equal(tf.argmax(self.out, 1), tf.argmax(label, 1))
        predication = tf.argmax(self.out, 1)
        accuracy = tf.reduce_mean(tf.cast(correct_predication, tf.float32))
        return loss, accuracy, optimize_loss, predication

    def train(self):
        for i in range(self.config.EPOCH):
            aver_acc = 0.0
            aver_loss = 0.0
            for j in range(self.config.BATCH_COUNT):
                image_batch, label_batch = self.data.return_batch_data(batch_size=self.config.BATCH_SIZE,
                                                                       index=j)
                loss, _, acc = self.sess.run(fetches=[self.loss, self.optimize_loss, self.accuracy],
                                             feed_dict={
                                                 self.input: image_batch,
                                                 self.label: label_batch,
                                                 self.keep_prob: 0.5,
                                                 self.is_training: True
                                             })
                aver_acc = (aver_acc * float(j) + acc) / (j + 1)
                aver_loss = (aver_loss * float(j) + loss) / (j + 1)
                print("Epoch %3d, Iter %3d, loss %.3lf, aver loss %.3lf, acc %.3lf, aver acc %.3lf"
                      % (i, j, loss, aver_loss, acc, aver_loss))
            self.save_model(model_path=self.model_dir, epoch=i)

    def eval_tensor(self, tensor, image_batch, keep_prob, label=None):
        if not label:
            label = np.zeros([image_batch.shape[0]])
        res = self.sess.run(fetches=[tensor],
                            feed_dict={
                                self.input: image_batch,
                                self.label: label,
                                self.keep_prob: keep_prob,
                                self.is_training: False
                            })
        return res

    def test(self):
        image_batch, label_batch = self.data.return_batch_data(batch_size=self.config.BATCH_SIZE,
                                                               index=0)
        loss, pred, acc = self.sess.run(fetches=[self.loss, self.predication, self.accuracy],
                                        feed_dict={
                                            self.input: image_batch,
                                            self.label: label_batch,
                                            self.keep_prob: 0.5
                                        })
        print(label_batch, pred)
        print('Loss = %.3lf, Acc = %.3lf' % (loss, acc))


if __name__ == '__main__':
    from src.model.mnistCNN.mnistConfig import MnistCNNConfig
    from src.data.mnistCNN.mnistCNNData import MnistCNNData
    from src.data.mnistCNN.mnistCNNDataConfig import MnistCNNDataConfig as dataConfig

    from dataset import DATASET_PATH

    sess = tf.InteractiveSession()
    config = dataConfig()
    data = MnistCNNData(data_path=DATASET_PATH + '/mnist/', config=config)
    config = MnistCNNConfig()
    model = MnistCNN(config=config, sess=sess, data=data)
    model.load_model(model_path=LOG_PATH + '/mnist/7-18-18-34-45/model/', epoch=3)
    model.test()
