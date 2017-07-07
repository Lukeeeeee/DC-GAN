import datetime
import os

from log import LOG_PATH
from src.model.model import Model


class DeepGAN(Model):
    def __init__(self, sess, discriminator, generator, data=None):
        super(DeepGAN, self).__init__(sess=sess, data=data)

        ti = datetime.datetime.now()
        self.log_dir = (LOG_PATH + '/' + str(ti.month) + '-' + str(ti.day) + '-' + str(ti.hour) + '-' + str(ti.minute)
                        + '-' + str(ti.second) + '/')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.model_dir = self.log_dir + 'model/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.log_dir)

        self.D = discriminator
        self.G = generator

    def train(self):
        pass

    def test(self):
        pass

    def save_model(self, model_path, epoch):
        pass

    def load_model(self, model_path):
        pass
