import math

from src.config import Config


class CatDiscriminatorConfig(Config):
    IN_HEIGHT = 500
    IN_WIDTH = 28
    IN_CHANNEL = 1

    LEARNING_RATE = 0.001

    CONV_LAYER_COUNT = 3

    CONV_LAYER_1_OUT_CHANNEL = 32

    CONV_LAYER_2_OUT_CHANNEL = 32

    CONV_LAYER_3_OUT_CHANNEL = 64

    CONV_STRIDE = 2

    FILTER_SIZE = 4

    VARIABLE_RANDOM_STANDARD_DEVIATION = 0.02

    BATCH_NORM_MEAN = 1.0

    BATCH_STANDARD_DEVIATION = 0.02

    CONV_OUT_HEIGHT = int(math.ceil(float(IN_HEIGHT) / float(math.pow(CONV_STRIDE, CONV_LAYER_COUNT))))
    CONV_OUT_WIDTH = int(math.ceil(float(IN_WIDTH) / float(math.pow(CONV_STRIDE, CONV_LAYER_COUNT))))

    OUTPUT_SIZE = 2

    PREFIX = 'CAT_DISCRIMINATOR_'

    # D_VARIABLE_DICT = {
    #     "W_1": tf.Variable(tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, IN_CHANNEL, CONV_LAYER_1_OUT_CHANNEL],
    #                                            stddev=VARIABLE_RANDOM_STANDARD_DEVIATION), name='W_1'),
    #     "B_1": tf.Variable(tf.constant(0.0, shape=[CONV_LAYER_1_OUT_CHANNEL]), name='B_1'),
    #     'BETA_1': tf.Variable(tf.constant(0.0, shape=[CONV_LAYER_1_OUT_CHANNEL]), name='BETA_1'),
    #     'GAMMA_1': tf.Variable(tf.random_normal(shape=[CONV_LAYER_1_OUT_CHANNEL], mean=BATCH_NORM_MEAN,
    #                                             stddev=BATCH_STANDARD_DEVIATION), name='GAMMA_1'),
    #
    #     "W_2": tf.Variable(tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, CONV_LAYER_1_OUT_CHANNEL,
    #                     CONV_LAYER_2_OUT_CHANNEL], stddev=VARIABLE_RANDOM_STANDARD_DEVIATION), name='W_2'),
    #     "B_2": tf.Variable(tf.constant(0.0, shape=[CONV_LAYER_2_OUT_CHANNEL]), name='B_2'),
    #     'BETA_2': tf.Variable(tf.constant(0.0, shape=[CONV_LAYER_2_OUT_CHANNEL]), name='BETA_2'),
    #     'GAMMA_2': tf.Variable(tf.random_normal(shape=[CONV_LAYER_2_OUT_CHANNEL], mean=BATCH_NORM_MEAN,
    #                                             stddev=BATCH_STANDARD_DEVIATION), name='GAMMA_2'),
    #
    #     "W_3": tf.Variable(tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, CONV_LAYER_2_OUT_CHANNEL,
    #                     CONV_LAYER_3_OUT_CHANNEL], stddev=VARIABLE_RANDOM_STANDARD_DEVIATION), name='W_3'),
    #     "B_3": tf.Variable(tf.constant(0.0, shape=[CONV_LAYER_3_OUT_CHANNEL]), name='B_3'),
    #     'BETA_3': tf.Variable(tf.constant(0.0, shape=[CONV_LAYER_3_OUT_CHANNEL]), name='BETA_3'),
    #     'GAMMA_3': tf.Variable(tf.random_normal(shape=[CONV_LAYER_3_OUT_CHANNEL], mean=BATCH_NORM_MEAN,
    #                                             stddev=BATCH_STANDARD_DEVIATION), name='GAMMA_3'),
    #
    #     "W_4": tf.Variable(tf.truncated_normal([CONV_OUT_HEIGHT * CONV_OUT_WIDTH * CONV_LAYER_3_OUT_CHANNEL, 1],
    #                                            stddev=VARIABLE_RANDOM_STANDARD_DEVIATION), name='W_4'),
    #     "B_4": tf.Variable(tf.constant(0.0, shape=[1]), name='b_4')
    # }

    @staticmethod
    def save_to_json(conf):
        return {
            conf.PREFIX + 'IN_HEIGHT': conf.IN_HEIGHT,
            conf.PREFIX + 'IN_WIDTH': conf.IN_WIDTH,
            conf.PREFIX + 'IN_CHANNEL': conf.IN_CHANNEL,
            conf.PREFIX + 'LEARNING_RATE': conf.LEARNING_RATE,
            conf.PREFIX + 'CONV_LAYER_COUNT': conf.CONV_LAYER_COUNT,
            conf.PREFIX + 'CONV_LAYER_1_OUT_CHANNEL': conf.CONV_LAYER_1_OUT_CHANNEL,
            conf.PREFIX + 'CONV_LAYER_2_OUT_CHANNEL': conf.CONV_LAYER_2_OUT_CHANNEL,
            conf.PREFIX + 'CONV_LAYER_3_OUT_CHANNEL': conf.CONV_LAYER_3_OUT_CHANNEL,
            conf.PREFIX + 'CONV_STRIDE': conf.CONV_STRIDE,
            conf.PREFIX + 'FILTER_SIZE': conf.FILTER_SIZE,
            conf.PREFIX + 'VARIABLE_RANDOM_STANDARD_DEVIATION': conf.VARIABLE_RANDOM_STANDARD_DEVIATION,
            conf.PREFIX + 'BATCH_NORM_MEAN': conf.BATCH_NORM_MEAN,
            conf.PREFIX + 'BATCH_STANDARD_DEVIATION': conf.BATCH_STANDARD_DEVIATION,
            conf.PREFIX + 'CONV_OUT_HEIGHT': conf.CONV_OUT_HEIGHT,
            conf.PREFIX + 'CONV_OUT_WIDTH': conf.CONV_OUT_WIDTH,
            conf.PREFIX + 'OUTPUT_SIZE': conf.OUTPUT_SIZE
        }


if __name__ == '__main__':
    a = CatDiscriminatorConfig()
    a.log_config('1.json')
