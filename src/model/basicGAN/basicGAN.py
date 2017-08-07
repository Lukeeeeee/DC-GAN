import datetime
import json
import os

import tensorflow as tf

from log import LOG_PATH
from src.model.basicGAN.Discriminator.discriminator import Discriminator
from src.model.basicGAN.Discriminator.discriminatorConfig import DiscriminatorConfig as Dconfig
from src.model.basicGAN.Generator.generator import Generator
from src.model.basicGAN.Generator.generatorConfig import GeneratorConfig as Gconfig
from src.model.basicGAN.Generator.step2Generator import Step2Generator
from src.model.basicGAN.Generator.singleConvGenerator import SingleConvGenerator
from src.model.model import Model
import numpy as np
from PIL import Image


class BasicGAN(Model):
    def __init__(self, sess, data, config, g_config=None, d_config=None, step2_flag=False, single_flag=False):
        super(BasicGAN, self).__init__(sess=sess, data=data, config=config)

        ti = datetime.datetime.now()
        self.log_dir = (LOG_PATH + '/' + str(ti.month) + '-' + str(ti.day) + '-' + str(ti.hour) + '-' + str(ti.minute)
                        + '-' + str(ti.second) + '/')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        config_log = [config.save_to_json(config),
                      g_config.save_to_json(g_config),
                      d_config.save_to_json(d_config),
                      data.config.save_to_json(data.config)
                      ]

        with open(self.log_dir + 'config.json', 'w') as f:
            json.dump(config_log, f, indent=4)

        self.model_dir = self.log_dir + 'model/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if g_config is None:
            g_config = Gconfig()
        if d_config is None:
            d_config = Dconfig()
        if step2_flag is True:
            self.G = Step2Generator(sess=sess, data=None, config=g_config)
        elif single_flag is True:
            self.G = SingleConvGenerator(sess=sess, data=None, config=g_config)
        else:
            self.G = Generator(sess=sess, data=None, config=g_config)
        self.D = Discriminator(sess=sess, data=None, generator=self.G, config=d_config)
        self.G.loss = self.D.generator_loss
        self.model_saver = tf.train.Saver()

        self.loss_log_list = []

        self.merged_summary = tf.summary.merge_all()
        # self.merged_summary = tf.summary.merge([self.G.loss_scalar_summary,
        #                                         self.D.loss_scalar_summary,
        #                                         self.D.accuracy_scalar_summary,
        #                                         self.D.real_accuracy_scalar_summary,
        #                                         self.D.fake_accuracy_scalar_summary])
        self.summary_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def train(self):
        count = 0
        for i in range(self.config.TRAINING_EPOCH):
            D_aver_acc = 0.0
            D_aver_loss = 0.0
            G_aver_loss = 0.0
            D_aver_fake_acc = 0.0
            D_aver_real_acc = 0.0
            for j in range(self.config.BATCH_COUNT):
                count = count + 1
                image_batch, z_batch = self.data.return_batch_data(batch_size=self.config.BATCH_SIZE,
                                                                   index=j)
                z_batch_2 = self.data.return_z_batch_data(batch_size=self.config.BATCH_SIZE,
                                                          index=self.config.BATCH_COUNT - j - 1)
                D_acc, D_loss, D_real_acc, D_fake_acc = self.update_discriminator(image_batch=image_batch,
                                                                                  z_batch=z_batch)

                G_loss_1 = self.update_generator(z_batch=z_batch)
                G_loss_2 = self.update_generator(z_batch=z_batch_2)

                self.run_summary(image_batch=image_batch, z_batch=z_batch, count=count)

                G_loss = (G_loss_1 + G_loss_2) / 2.0

                # G_loss = self.update_generator(z_batch=z_batch)

                D_aver_acc = (D_aver_acc * float(j) + D_acc) / float(j + 1)
                D_aver_loss = (D_aver_loss * float(j) + D_loss) / float(j + 1)

                G_aver_loss = (G_aver_loss * float(j) + G_loss) / float(j + 1)

                D_aver_fake_acc = (D_aver_fake_acc * float(j) + D_fake_acc) / float(j + 1)
                D_aver_real_acc = (D_aver_real_acc * float(j) + D_real_acc) / float(j + 1)

                # print(self.eval_tensor(tensor=self.D.real_D, image_batch=image_batch, z_batch=z_batch))
                # print(self.eval_tensor(tensor=self.D.fake_D, image_batch=image_batch, z_batch=z_batch))

                print(
                "Epoch %5d, Iter %5d: D acc %.3lf aver acc %.3lf Fake acc %.3lf, Real acc %.3lf, loss %.3lf aver loss %.3lf, G loss %.3lf, "
                "aver loss %.3lf" % (
                i, j, D_acc, D_aver_acc, D_aver_fake_acc, D_aver_real_acc, D_loss, D_aver_loss, G_loss, G_aver_loss))
                # self.run_summary(image_batch, z_batch, count)
            self.loss_log_list.append({
                'D_accuracy': D_aver_acc,
                'D_real_accuracy': D_aver_fake_acc,
                'D_fake_accuracy': D_aver_real_acc,
                'D_loss': D_aver_loss,
                'G_loss': G_aver_loss,
                'Epoch': i
            })
            if (i + 1) % self.config.SAVE_MODEL_EVERY_EPOCH == 0:
                self.save_model(model_path=self.model_dir, epoch=i + 1)
                # if (i + 1) % self.config.TEST_EVERY_EPOCH == 0:
                #     image, z = self.data.return_batch_data(self.config.BATCH_SIZE, 0)
                #     res = self.eval_tensor(tensor=self.G.output, image_batch=image, z_batch=z)
                #     for i in range(5):
                #         data = np.multiply(np.add(res, 1.0), 127.5)
                #         data = np.reshape(data[i:i + 1, ],
                #                           newshape=[28, 28, 1],
                #                           ).astype((np.uint8))
                #         im = Image.fromarray(data)
                #         im.show()
                #     for i in range(5):
                #         # data = np.multiply(np.add(image, 1.0), 127.5)
                #         data = np.reshape(image[i:i + 1, ],
                #                           newshape=[224, 224, 3],
                #                           ).astype((np.uint8))
                #         im = Image.fromarray(data, mode='RGB')
                #         im.show()

        with open(self.log_dir + 'loss.json', 'w') as f:
            json.dump(self.loss_log_list, f, indent=4)

    def test(self):
        pass

    def update_generator(self, z_batch):
        loss, _, out = self.sess.run(fetches=[self.G.loss, self.G.optimize_loss, self.G.output],
                                     feed_dict={self.G.input: z_batch,
                                                self.G.is_training: True,
                                                self.D.is_training: True})
        return loss

    def update_discriminator(self, image_batch, z_batch):
        acc, loss, real_acc, fake_acc, _ = self.sess.run(fetches=[self.D.accuracy,
                                                                  self.D.loss,
                                                                  self.D.real_accuracy,
                                                                  self.D.fake_accuracy,
                                                                  self.D.minimize_loss],
                                                         feed_dict={self.D.input: image_batch,
                                                                    self.G.input: z_batch,
                                                                    self.G.is_training: True,
                                                                    self.D.is_training: True})
        return acc, loss, real_acc, fake_acc

    def run_summary(self, image_batch, z_batch, count):
        summary = self.sess.run(fetches=self.merged_summary,
                                feed_dict={self.D.input: image_batch,
                                           self.G.input: z_batch,
                                           self.G.is_training: False,
                                           self.D.is_training: False})
        self.summary_writer.add_summary(summary, count)

    def eval_tensor(self, tensor, image_batch, z_batch):
        res = self.sess.run(tensor, feed_dict={self.D.input: image_batch,
                                               self.G.input: z_batch,
                                               self.G.is_training: False,
                                               self.D.is_training: False})
        return res

    def save_model(self, model_path, epoch):
        self.model_saver.save(sess=self.sess,
                              save_path=model_path + 'model.ckpt',
                              global_step=epoch)
        print('Model saved at %s' % model_path)

    def load_model(self, model_path, epoch):
        self.model_saver.restore(sess=self.sess,
                                 save_path=model_path + 'model.ckpt-' + str(epoch))
        print('Model loaded at %s' % model_path)

    def log_config(self):
        with open(self.log_dir + 'Model.json', 'w') as f:

            gan_dict = self.config.save_to_json(self.config)
            d_dict = Dconfig.save_to_json(Dconfig)
            g_dict = Gconfig.save_to_json(Gconfig)

            dict_list = [gan_dict, d_dict, g_dict]

            log_dict = {}

            for dict_ in dict_list:
                for key, value in dict_.iteritems():
                    log_dict[str(key)] = value
            json.dump(log_dict, f, indent=4)

        with open(self.log_dir + 'Data.json', 'w') as f:
            data_dict = self.data.config.save_to_json(self.data.config)
            json.dump(data_dict, f, indent=4)
