import datetime
import os

import tensorflow as tf

from log import LOG_PATH
from src.model.deepGAN.Discriminator.discriminator import Discriminator
from src.model.deepGAN.Generator.generator import Generator
from src.model.deepGAN.ganConfig import GANConfig as ganConfig
from src.model.model import Model


class DeepGAN(Model):
    def __init__(self, sess, data=None):
        super(DeepGAN, self).__init__(sess=sess, data=data)

        ti = datetime.datetime.now()
        self.log_dir = (LOG_PATH + '/' + str(ti.month) + '-' + str(ti.day) + '-' + str(ti.hour) + '-' + str(ti.minute)
                        + '-' + str(ti.second) + '/')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.model_dir = self.log_dir + 'model/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.G = Generator(sess=sess, data=None)
        self.D = Discriminator(sess=sess, data=None, generator=self.G)
        self.G.loss = self.D.generator_loss
        self.model_saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

    def train(self):
        for i in range(ganConfig.TRAINING_EPOCH):
            D_aver_acc = 0.0
            D_aver_loss = 0.0
            G_aver_loss = 0.0
            for j in range(ganConfig.BATCH_COUNT):
                image_batch, z_batch = self.data.return_batch_data(batch_size=ganConfig.BATCH_SIZE,
                                                                   index=j)
                D_acc, D_loss = self.update_discriminator(image_batch=image_batch, z_batch=z_batch)
                G_loss_1 = self.update_generator(z_batch=z_batch)
                G_loss_2 = self.update_generator(z_batch=z_batch)
                G_loss = (G_loss_1 + G_loss_2) / 2.0

                D_aver_acc = (D_aver_acc * float(j) + D_acc) / float(j + 1)
                D_aver_loss = (D_aver_loss * float(j) + D_loss) / float(j + 1)

                G_aver_loss = (G_aver_loss * float(j) + G_loss) / float(j + 1)
                print("Epoch %5d, Iter %5d: D acc %.3lf aver acc %.3lf, loss %.3lf aver loss %.3lf, G loss %.3lf, "
                      "aver loss %.3lf" % (i, j, D_acc, D_aver_acc, D_loss, D_aver_loss, G_loss, G_aver_loss))

    def test(self):
        pass

    def update_generator(self, z_batch):
        loss, _ = self.sess.run(fetches=[self.G.loss, self.G.optimize_loss],
                                feed_dict={self.G.input: z_batch,
                                           self.G.is_training: True,
                                           self.D.is_training: True})
        return loss

    def update_discriminator(self, image_batch, z_batch):
        acc, loss, _ = self.sess.run(fetches=[self.D.accuracy,
                                              self.D.loss,
                                              self.D.minimize_loss],
                                     feed_dict={self.D.input: image_batch,
                                                self.G.input: z_batch,
                                                self.G.is_training: True,
                                                self.D.is_training: True})
        return acc, loss

    def save_model(self, model_path, epoch):
        self.model_saver.save(sess=self.sess,
                              save_path=model_path,
                              global_step=epoch)
        print('Model saved at %s' % model_path)

    def load_model(self, model_path):
        self.model_saver.restore(sess=self.sess,
                                 save_path=model_path)
        print('Model loaded at %s' % model_path)
