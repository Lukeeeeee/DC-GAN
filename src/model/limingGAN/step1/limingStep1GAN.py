from src.model.model import Model
import tensorflow as tf


class LimingStep1GAN(Model):
    def __init__(self, sess, config, data):
        super(LimingStep1GAN, self).__init__(sess=sess, config=config, data=data)
        self.name = 'LIMING_STEP1_GAN'

        # self.noise_input = tf.placeholder(tf.float32, shape=[None, Noise_h, Noise_w, Noise_ch], name='noise')
        # self.noise_sample_input = tf.placeholder(tf.float32, shape=[None, Noise_h, Noise_w, Noise_ch], name='noise')
        # # real data input
        # self.image_input = tf.placeholder(tf.float32, shape=[None, Image_h, Image_w, Image_ch], name='image')
        #
