# dcgan by liming @17.7.10
import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = CURRENT_PATH + '/../../'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

sys.path.append(CURRENT_PATH)
sys.path.append(PROJECT_PATH)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import numpy as np
from PIL import Image
from glob import glob
import numpy
import scipy.stats as stats
from model import *
from pre_data import *
from log import LOG_PATH
import datetime

Image_h = 7
Image_w = 7
Image_ch = 16
Noise_h = 1
Noise_w = 1
Noise_ch = 100
Epoch_num = 5
Batch_size = 200
Sample_num = 10000
G_learnrate = 1e-3
D_learnrate = 1e-3
ti = datetime.datetime.now()
z_data_save_dir = (
LOG_PATH + '/liming_test/' + str(ti.month) + '-' + str(ti.day) + '-' + str(ti.hour) + '-' + str(ti.minute)
+ '-' + str(ti.second) + '/')
if not os.path.exists(z_data_save_dir):
    os.makedirs(z_data_save_dir)
log_dir = LOG_PATH + '/liming_step1/8-13-20-14-22'
model_dir = log_dir + '/model/'
model_epoch = 343


def optimizer(loss, learning_rate, vlist=None, name=None):
    with tf.variable_scope(name):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name=name + '/Adam')
        return opt.minimize(loss, var_list=vlist, name=name + '/opt')


def draw_img(x):
    pass


def __main__():
    # noise input
    noise_input = tf.placeholder(tf.float32, shape=[None, Noise_h, Noise_w, Noise_ch], name='noise')
    # real data input
    image_input = tf.placeholder(tf.float32, shape=[None, Image_h, Image_w, Image_ch], name='image')

    # generate G
    G = generate(z=noise_input, h=Image_h, w=Image_w, ch=Image_ch, is_training=True, reuse=None, batch_size=Batch_size)

    img_sample = restruct_image(G, Batch_size)
    tf.summary.image('generated image', img_sample, Batch_size)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # =====tensorboard=============

        # =============================
        sess.run(tf.global_variables_initializer())
        saver.restore(sess=sess, save_path=model_dir + '/./model.ckpt-' + str(model_epoch))

        # mean = 0.8
        # scale = 0.01
        # z = stats.truncnorm.rvs((0.0 - mean) / scale, (1 - mean) / scale, loc=mean, scale=scale, size=Batch_size * 100)
        # z = np.reshape(np.array(z), newshape=[Batch_size, 1, 1, 100])

        z = np.random.uniform(low=0.0, high=1.0, size=[Batch_size, 1, 1, 100])

        res = sess.run(G, feed_dict={
            noise_input: z,
        })
        np.save(file=z_data_save_dir + '/7_7_16.npy', arr=res)


# =================================================================
if __name__ == "__main__":
    __main__()
