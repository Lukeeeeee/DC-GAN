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

from model import decrim, generate
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


def __main__():
    # noise input
    save_log(log_dir)
    noise_input = tf.placeholder(tf.float32, shape=[None, Noise_h, Noise_w, Noise_ch], name='noise')
    noise_sample_input = tf.placeholder(tf.float32, shape=[None, Noise_h, Noise_w, Noise_ch], name='noise')
    # real data input
    image_input = tf.placeholder(tf.float32, shape=[None, Image_h, Image_w, Image_ch], name='image')

    # generate G
    G = generate(z=noise_input, h=Image_h, w=Image_w, ch=Image_ch, is_training=True, reuse=None, batch_size=Batch_size)
    # param of G
    G_vars = tf.trainable_variables()

    G_sample = generate(z=noise_sample_input, h=Image_h, w=Image_w, ch=Image_ch, is_training=False, reuse=True,
                        batch_size=Batch_size)
    # img_sample = restruct_image(G_sample, Batch_size)
    # tf.summary.image('generated image', img_sample, Batch_size)
    # decrim
    D = decrim(image_input, True, None, batch_size=Batch_size)
    # param of d
    D_vars = []
    for item in tf.trainable_variables():
        if item not in G_vars:
            D_vars.append(item)

    d_real = D
    d_fake = decrim(G, True, True)

    loss_train_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))) \
                   + tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
    tf.summary.scalar('d_loss', loss_train_D)

    loss_train_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
    tf.summary.scalar('g_loss', loss_train_G)
    # loss_train_G = d_fake
    # loss_train_D = -(d_real + d_fake)
    # loss_train_G = (1 / 2) * (d_fake - 1) ** 2
    # loss_train_D = (1 / 2) * (d_real - 1) ** 2 + (1 / 2) * (d_fake) ** 2
    g_optimizer = optimizer(loss_train_G, G_learnrate, G_vars, name='opt_train_G')
    d_optimizer = optimizer(loss_train_D, D_learnrate, D_vars, name='opt_train_D')

    saver = tf.train.Saver(max_to_keep=50)
    # noise_sample = np.random.uniform(-1,1,[Batch_size,100]).astype('float32')
    # ==============================Start training=============================
    with tf.Session() as sess:
        # =====tensorboard=============
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(tensorboad_dir, sess.graph)
        # =============================
        sess.run(tf.global_variables_initializer())
        image_data, z_data = get_datalist()
        image_len = Sample_num
        batch_num = int(image_len / Batch_size)
        count = 0
        for e in range(Epoch_num):

            for idx in range(batch_num):

                # prepare data
                # TODO z data
                img_batch = image_data[idx * Batch_size: (idx + 1) * Batch_size, ]
                z = z_data[idx * Batch_size: (idx + 1) * Batch_size, ]

                _, d_loss = sess.run([d_optimizer, loss_train_D],
                                     feed_dict={
                                         noise_input: z,
                                         image_input: img_batch,

                                     })

                _, g_loss_1 = sess.run([g_optimizer, loss_train_G],
                                       feed_dict={
                                           noise_input: z,
                                           image_input: img_batch,

                                       })

                # _, g_loss_2 = sess.run([g_optimizer, loss_train_G],
                #                        feed_dict={
                #                            noise_input: z,
                #                            image_input: img_batch,
                #
                #                        })
                g_loss = g_loss_1

                print("epoch: %d batch: %d  gloss:%.4f dloss:%.4f" %
                      (e + 1, idx, g_loss, d_loss))

                if idx % 10 == 0:
                    noise_sample = z_data[0 * Batch_size: 1 * Batch_size, ]
                    sumarry_all = sess.run(merged_summary_op, feed_dict={
                        noise_sample_input: noise_sample,
                        noise_input: z,
                        image_input: img_batch,
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
