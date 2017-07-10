import tensorflow as tf

from src.common.ops import ops
from src.data.dataConfig import DataConfig
from src.model.deepGAN.Generator.generatorConfig import GeneratorConfig as g_config
from src.model.model import Model


class Generator(Model):
    def __init__(self, sess, data):
        super(Generator, self).__init__(sess=sess, data=data)
        self.name = 'Generator'
        with tf.variable_scope('Discriminator'):
            self.variable_dict = {
                'W_1': tf.Variable(tf.truncated_normal(shape=([g_config.IN_CHANNEL,
                                                               g_config.TRAN_CONV_LAYER_1_IN_CHANNEL *
                                                               g_config.TRAN_CONV_LAYER_1_HEIGHT *
                                                               g_config.TRAN_CONV_LAYER_1_WIDTH]),
                                                       stddev=g_config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                                   name='W_1'),
                'B_1': tf.Variable(tf.constant(value=0.0,
                                               shape=[g_config.TRAN_CONV_LAYER_1_IN_CHANNEL *
                                                      g_config.TRAN_CONV_LAYER_1_WIDTH *
                                                      g_config.TRAN_CONV_LAYER_1_HEIGHT]),
                                   name='B_1'),
                'BETA_1': tf.Variable(tf.truncated_normal(shape=[g_config.TRAN_CONV_LAYER_1_IN_CHANNEL]),
                                      name='BETA_1'),

                'GAMMA_1': tf.Variable(tf.random_normal(shape=[g_config.TRAN_CONV_LAYER_1_IN_CHANNEL],
                                                        mean=g_config.BATCH_NORM_MEAN,
                                                        stddev=g_config.BATCH_STANDARD_DEVIATION),
                                       name='GAMMA_1'),

                'W_2': tf.Variable(tf.truncated_normal(shape=[g_config.FILTER_SIZE,
                                                              g_config.FILTER_SIZE,
                                                              g_config.TRAN_CONV_LAYER_2_IN_CHANNEL,
                                                              g_config.TRAN_CONV_LAYER_1_IN_CHANNEL],
                                                       stddev=g_config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                                   name='W_2'),

                'B_2': tf.Variable(tf.constant(value=0.0,
                                               shape=[g_config.TRAN_CONV_LAYER_2_IN_CHANNEL]),
                                   name='B_2'),

                'BETA_2': tf.Variable(tf.constant(value=0.0,
                                                  shape=[g_config.TRAN_CONV_LAYER_2_IN_CHANNEL]),
                                      name='BETA_2'),
                'GAMMA_2': tf.Variable(tf.random_normal(shape=[g_config.TRAN_CONV_LAYER_2_IN_CHANNEL]),
                                       name='GAMMA_2'),

                'W_3': tf.Variable(tf.truncated_normal(shape=[g_config.FILTER_SIZE,
                                                              g_config.FILTER_SIZE,
                                                              g_config.TRAN_CONV_LAYER_3_IN_CHANNEL,
                                                              g_config.TRAN_CONV_LAYER_2_IN_CHANNEL],
                                                       stddev=g_config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                                   name='W_3'),

                'B_3': tf.Variable(tf.constant(value=0.0,
                                               shape=[g_config.TRAN_CONV_LAYER_3_IN_CHANNEL]),
                                   name='B_3'),

                'BETA_3': tf.Variable(tf.constant(value=0.0,
                                                  shape=[g_config.TRAN_CONV_LAYER_3_IN_CHANNEL]),
                                      name='BETA_3'),

                'GAMMA_3': tf.Variable(tf.random_normal(shape=[g_config.TRAN_CONV_LAYER_3_IN_CHANNEL]),
                                       name='GAMMA_3'),

                'W_4': tf.Variable(tf.truncated_normal(shape=[g_config.FILTER_SIZE,
                                                              g_config.FILTER_SIZE,
                                                              g_config.TRAN_CONV_LAYER_3_OUT_CHANNEL,
                                                              g_config.TRAN_CONV_LAYER_3_IN_CHANNEL],
                                                       stddev=g_config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                                   name='W_4'),
                'B_4': tf.Variable(tf.constant(value=0.0,
                                               shape=[g_config.TRAN_CONV_LAYER_3_OUT_CHANNEL]),
                                   name='B_4'),

                'BETA_4': tf.Variable(tf.constant(value=0.0,
                                                  shape=[g_config.TRAN_CONV_LAYER_3_OUT_CHANNEL]),
                                      name='BETA_4'),

                'GAMMA_4': tf.Variable(tf.random_normal(shape=[g_config.TRAN_CONV_LAYER_3_OUT_CHANNEL],
                                                        mean=g_config.BATCH_NORM_MEAN,
                                                        stddev=g_config.BATCH_STANDARD_DEVIATION),
                                       name='GAMMA_4')

            }
        with tf.variable_scope('Generator'):
            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=[None, g_config.IN_WIDTH, g_config.IN_HEIGHT, g_config.IN_CHANNEL],
                                        name='INPUT')
            self.loss = None

    def set_loss(self, new_loss):
        self.loss = new_loss

    def set_input(self, new_input):
        self.input = new_input

    def create_model(self):

        pass

    def create_training_method(self):
        pass

    def generate(self):
        pass

