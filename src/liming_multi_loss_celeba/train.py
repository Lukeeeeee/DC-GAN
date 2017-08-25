# dcgan by liming @17.7.10
import os
import sys
import json
import tensorflow as tf
from log import LOG_PATH
import datetime

Image_h = 224
Image_w = 224
Image_ch = 3

Pool_1_h = 112
Pool_1_w = 112
Pool_1_ch = 64

Pool_2_h = 56
Pool_2_w = 56
Pool_2_ch = 128

Pool_3_h = 28
Pool_3_w = 28
Pool_3_ch = 256

Pool_4_h = 14
Pool_4_w = 14
Pool_4_ch = 512

Pool_5_h = 7
Pool_5_w = 7
Pool_5_ch = 512

Noise_h = 1
Noise_w = 1
Noise_ch = 30

Epoch_num = 500
Batch_size = 200
Sample_num = 60000
G_learnrate = 1e-3
D_learnrate = 1e-3

ti = datetime.datetime.now()
log_dir = (
    LOG_PATH + '/liming_multi_loss/' + str(ti.month) + '-' + str(ti.day) + '-' + str(ti.hour) + '-' + str(ti.minute)
    + '-' + str(ti.second) + '/')
tensorboad_dir = log_dir

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = CURRENT_PATH + '/../../'
sys.path.append(CURRENT_PATH)
sys.path.append(PROJECT_PATH)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from model import decrim_list, generate
from model import save_log as model_save_log
from pre_data import get_datalist, restruct_image

model_dir = log_dir + '/model/'


def save_log(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    json_log = {
        "Image_h": Image_h,
        "Image_w": Image_w,
        "Image_ch": Image_ch,
        "Noise_h": Noise_h,
        "Noise_w": Noise_w,
        "Noise_ch": Noise_ch,

        "Epoch_num": Epoch_num,
        "Batch_size": Batch_size,
        "Sample_num": Sample_num,
        "G_learnrate": G_learnrate,
        "D_learnrate": D_learnrate,

    }
    with open(log_dir + '/train_config.json', 'w') as f:
        json.dump(json_log, f, indent=4)
    model_save_log(log_dir)


def optimizer(loss, learning_rate, vlist=None, name=None):
    with tf.variable_scope(name):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name=name + '/Adam')
        return opt.minimize(loss, var_list=vlist, name=name + '/opt')


def draw_img(x):
    pass


# output 224 224 3
# 112 112 64
# 56 56 128
# 28 28 256
# 14 14 512
# 7 7 512


def __main__():
    # save_log(log_dir)

    noise_input = tf.placeholder(tf.float32, shape=[None, Noise_h, Noise_w, Noise_ch], name='noise')
    noise_sample_input = tf.placeholder(tf.float32, shape=[None, Noise_h, Noise_w, Noise_ch], name='noise')

    image_input = tf.placeholder(tf.float32, shape=[None, Image_h, Image_w, Image_ch], name='image')
    image_112_112_64 = tf.placeholder(tf.float32, shape=[None, 112, 112, 64], name='image112_112_64')
    image_56_56_128 = tf.placeholder(tf.float32, shape=[None, 56, 56, 128], name='image56_56_128')
    image_28_28_256 = tf.placeholder(tf.float32, shape=[None, 28, 28, 256], name='image28_28_256')
    image_14_14_512 = tf.placeholder(tf.float32, shape=[None, 14, 14, 512], name='image_14_14_512')

    image_input_list = [image_input, image_112_112_64, image_56_56_128, image_28_28_256, image_14_14_512]
    G_conv_list = generate(z=noise_input, h=Image_h, w=Image_w, ch=Image_ch, is_training=True, reuse=None,
                           batch_size=Batch_size)
    G_vars = tf.trainable_variables()

    G_sample = generate(z=noise_sample_input, h=Image_h, w=Image_w, ch=Image_ch, is_training=False, reuse=True,
                        batch_size=Batch_size)
    img_sample = restruct_image(G_sample[0], Batch_size)
    tf.summary.image('generated image', img_sample, Batch_size)

    D_list = decrim_list(image_input_list, True, None)
    D_vars = []
    for item in tf.trainable_variables():
        if item not in G_vars:
            D_vars.append(item)

    d_real_list = D_list
    d_fake_list = decrim_list(G_conv_list, True, True)
    d_loss_list = []

    for d_real, d_fake in zip(d_real_list, d_fake_list):
        loss_train_D = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))) \
                       + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        d_loss_list.append(loss_train_D)
        tf.summary.scalar('d_loss', loss_train_D)

    d_loss = tf.add_n(d_loss_list)
    tf.summary.scalar('d_loss_sum', d_loss)

    g_loss_list = []

    for d_fake in d_fake_list:
        loss_train_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        tf.summary.scalar('g_loss', loss_train_G)
        g_loss_list.append(loss_train_G)
    g_loss = tf.add_n(g_loss_list)
    tf.summary.scalar('g_loss_sum', g_loss)

    g_optimizer = optimizer(g_loss, G_learnrate, G_vars, name='opt_train_G')
    d_optimizer = optimizer(d_loss, D_learnrate, D_vars, name='opt_train_D')

    saver = tf.train.Saver(max_to_keep=50)

    with tf.Session() as sess:
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(tensorboad_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        image_data, image_14_data, image_28_data, image_56_data, image_112_data, z_data = get_datalist()
        image_len = Sample_num
        batch_num = int(image_len / Batch_size)
        count = 0
        for e in range(Epoch_num):

            for idx in range(batch_num):

                img_batch = image_data[idx * Batch_size: (idx + 1) * Batch_size, ]
                img_14_batch = image_14_data[idx * Batch_size: (idx + 1) * Batch_size, ]
                img_28_batch = image_28_data[idx * Batch_size: (idx + 1) * Batch_size, ]
                img_56_batch = image_56_data[idx * Batch_size: (idx + 1) * Batch_size, ]
                img_112_batch = image_112_data[idx * Batch_size: (idx + 1) * Batch_size, ]

                z = z_data[idx * Batch_size: (idx + 1) * Batch_size, ]

                _, d_loss_res = sess.run([d_optimizer, d_loss],
                                         feed_dict={
                                             noise_input: z,
                                             image_input: img_batch,
                                             image_14_14_512: img_14_batch,
                                             image_28_28_256: img_28_batch,
                                             image_56_56_128: img_56_batch,
                                             image_112_112_64: img_112_batch
                                         })

                _, g_loss_res = sess.run([g_optimizer, g_loss],
                                         feed_dict={
                                             noise_input: z,
                                             image_input: img_batch,
                                             image_14_14_512: img_14_batch,
                                             image_28_28_256: img_28_batch,
                                             image_56_56_128: img_56_batch,
                                             image_112_112_64: img_112_batch

                                         })

                # _, g_loss_2 = sess.run([g_optimizer, loss_train_G],
                #                        feed_dict={
                #                            noise_input: z,
                #                            image_input: img_batch,
                #
                #                        })

                print("epoch: %d batch: %d  gloss:%.4f dloss:%.4f" %
                      (e + 1, idx, g_loss_res, d_loss_res))

                if idx % 10 == 0:
                    noise_sample = z_data[0 * Batch_size: 1 * Batch_size, ]
                    sumarry_all = sess.run(merged_summary_op, feed_dict={
                        noise_sample_input: noise_sample,
                        noise_input: z,
                        image_input: img_batch,
                        image_14_14_512: img_14_batch,
                        image_28_28_256: img_28_batch,
                        image_56_56_128: img_56_batch,
                        image_112_112_64: img_112_batch
                    })
                    summary_writer.add_summary(sumarry_all, count)
                    count = count + 1
            if e % 4 == 0:
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                saver.save(sess=sess,
                           save_path=model_dir + '/model.ckpt',
                           global_step=e)


if __name__ == "__main__":
    __main__()
