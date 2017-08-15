# dcgan by liming @17.7.10
import os
import sys
import json
import tensorflow as tf
from log import LOG_PATH
import datetime
import numpy as np

log_dir = LOG_PATH + '/liming_step1/8-15-20-37-20'
model_epoch = 448


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = CURRENT_PATH + '/../../'
sys.path.append(CURRENT_PATH)
sys.path.append(PROJECT_PATH)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from model import generate
from pre_data import restruct_image
ti = datetime.datetime.now()
z_data_save_dir = (
LOG_PATH + '/liming_test/' + str(ti.month) + '-' + str(ti.day) + '-' + str(ti.hour) + '-' + str(ti.minute)
+ '-' + str(ti.second) + '/')
model_dir = log_dir + '/model/'


def optimizer(loss, learning_rate, vlist=None, name=None):
    with tf.variable_scope(name):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name=name + '/Adam')
        return opt.minimize(loss, var_list=vlist, name=name + '/opt')


def __main__():
    # noise input
    with open(log_dir + '/train_config.json') as f:
        train_config = json.load(f)
        Image_h = train_config['Image_h']
        Image_w = train_config['Image_w']
        Image_ch = train_config['Image_ch']
        Noise_h = train_config['Noise_h']
        Noise_w = train_config['Noise_w']
        Noise_ch = train_config['Noise_ch']
        Batch_size = train_config['Batch_size']

    noise_input = tf.placeholder(tf.float32, shape=[None, Noise_h, Noise_w, Noise_ch], name='noise')

    with open(log_dir + '/model_config.json') as f:
        model_config = json.load(f)

    G = generate(z=noise_input,
                 h=Image_h,
                 w=Image_w,
                 ch=Image_ch,
                 is_training=True,
                 reuse=None,
                 batch_size=Batch_size,
                 model_config=model_config,
                 train_config=train_config)

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

        z = np.random.uniform(low=0.0, high=1.0, size=[Batch_size, Noise_h, Noise_w, Noise_ch])

        res = sess.run(G, feed_dict={
            noise_input: z,
        })
        if not os.path.exists(z_data_save_dir):
            os.makedirs(z_data_save_dir)
        np.save(file=z_data_save_dir + '/7_7_16.npy', arr=res)


# =================================================================
if __name__ == "__main__":
    __main__()
