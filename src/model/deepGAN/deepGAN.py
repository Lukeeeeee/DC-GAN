import datetime
import os

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
            os.makedirs(self.log_dir)
        self.G = Generator(sess=sess, data=None)
        self.D = Discriminator(sess=sess, data=None, generator=self.G)
        self.G.loss = self.D.generator_loss

    def train(self):
        for i in range(ganConfig.TRAINING_EPOCH):
            for j in range(ganConfig.BATCH_COUNT):
                image_batch, z_batch = self.data.return_one_batch(batch_size=ganConfig.BATCH_SIZE,
                                                                  batch_index=j)
                D_acc, D_loss, D_gradients = self.D.update(image_batch=image_batch,
                                                           z_batch=z_batch)
                G_loss_1, G_gradients_1 = self.G.update(z_batch=z_batch)
                G_loss_2, G_gradients_2 = self.G.update(z_batch=z_batch)

    def test(self):
        pass

    def save_model(self, model_path, epoch):
        self.model_saver.save(sess=self.sess,
                              save_path=model_path,
                              global_step=epoch)
        print('Model saved at %s' % model_path)

    def load_model(self, model_path):
        self.model_saver.restore(sess=self.sess,
                                 save_path=model_path)
        print('Model loaded at %s' % model_path)
