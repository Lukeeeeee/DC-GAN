import os

from src.model.basicGAN.basicGAN import BasicGAN


class DeepGAN(BasicGAN):
    def __init__(self, sess, data, config, g_config=None, d_config=None):
        super(DeepGAN, self).__init__(sess=sess, data=data, config=config, g_config=g_config, d_config=d_config)
        self.log_dir = self.log_dir + '/' + self.config.NAME + '/'
        self.model_dir = self.log_dir + '/model/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
