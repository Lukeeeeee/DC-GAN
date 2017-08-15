# dcgan by liming @17.7.10
import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = CURRENT_PATH + '/../../'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

sys.path.append(CURRENT_PATH)
sys.path.append(PROJECT_PATH)
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from glob import glob
from model import *
from pre_data import *
from log import LOG_PATH
import datetime

Image_h = 28
Image_w = 28
Image_ch = 1
Noise_h = 7
Noise_w = 7
Noise_ch = 16
Epoch_num = 500
Sample_num = 10000
Batch_size = 200
G_learnrate = 1e-3
D_learnrate = 1e-3
ti = datetime.datetime.now()
log_dir = LOG_PATH + '/liming_test/8-15-21-18-16'
tensorboad_dir = log_dir
model_dir = LOG_PATH + '/liming_step2/8-15-16-41-7/model'
model_epoch = 200


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
    G = generate(noise_input, Image_h, Image_w, True, None, batch_size=Batch_size)
    # param of G
    img_sample = restruct_image(G, Batch_size)
    tf.summary.image('generated image', img_sample, Batch_size)
    saver = tf.train.Saver()

    # noise_sample = np.random.uniform(-1,1,[Batch_size,100]).astype('float32')
    # ==============================Start training=============================
    with tf.Session() as sess:
        # =====tensorboard=============
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(tensorboad_dir, sess.graph)
        # =============================
        sess.run(tf.global_variables_initializer())
        saver.restore(sess=sess, save_path=model_dir + '/model.ckpt-' + str(model_epoch))
        z = np.load(file=log_dir + '/7_7_16.npy')

        sumarry_all = sess.run(merged_summary_op, feed_dict={
            noise_input: z,
        })
        summary_writer.add_summary(sumarry_all, 0)


if __name__ == "__main__":
    __main__()
