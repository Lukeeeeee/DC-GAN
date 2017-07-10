import math


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class GeneratorConfig(object):

    IN_HEIGHT = 1
    IN_WIDTH = 1
    IN_CHANNEL = 100

    OUT_HEIGHT = 24
    OUT_WIDTH = 24
    OUT_CHANNEL = 256

    TRAN_CONV_LAYER_1_IN_CHANNEL = 1024
    TRAN_CONV_LAYER_2_IN_CHANNEL = 512
    TRAN_CONV_LAYER_3_IN_CHANNEL = 512
    TRAN_CONV_LAYER_3_OUT_CHANNEL = 256

    FILTER_SIZE = 4

    CONV_STRIDE = 2

    TRAN_CONV_LAYER_3_HEIGHT = int(math.ceil(float(OUT_HEIGHT) / float(CONV_STRIDE)))
    TRAN_CONV_LAYER_3_WIDTH = int(math.ceil(float(OUT_WIDTH) / float(CONV_STRIDE)))

    TRAN_CONV_LAYER_2_HEIGHT = int(math.ceil(float(TRAN_CONV_LAYER_3_HEIGHT) / float(CONV_STRIDE)))
    TRAN_CONV_LAYER_2_WIDTH = int(math.ceil(float(TRAN_CONV_LAYER_3_WIDTH) / float(CONV_STRIDE)))

    TRAN_CONV_LAYER_1_HEIGHT = int(math.ceil(float(TRAN_CONV_LAYER_2_HEIGHT) / float(CONV_STRIDE)))
    TRAN_CONV_LAYER_1_WIDTH = int(math.ceil(float(TRAN_CONV_LAYER_2_WIDTH) / float(CONV_STRIDE)))

    G_LEARNING_RATE = 0.003

    DATA_COUNT = 5000
    DATA_SOURCE = 'relu3_1'
    DATA_Z_NAME = None

    BATCH_SIZE = 200

    VARIABLE_RANDOM_STANDARD_DEVIATION = 0.02

    BATCH_NORM_MEAN = 1.0

    BATCH_STANDARD_DEVIATION = 0.02

    @staticmethod
    def save_to_json(conf):
        return {
            'IN_HEIGHT': conf.IN_HEIGHT,
            'IN_WIDTH': conf.IN_WIDTH,
            'IN_CHANNEL': conf.IN_CHANNEL,

            'OUT_HEIGHT': conf.OUT_HEIGHT,
            'OUT_WIDTH': conf.OUT_WIDTH,
            'OUT_CHANNEL': conf.OUT_CHANNEL,

            'G_LEARNING_RATE': conf.G_LEARNING_RATE,

            'DATA_COUNT': conf.DATA_COUNT,
            'DATA_SOURCE': conf.DATA_SOURCE,
            'DATA_Z_NAME': conf.DATA_Z_NAME,

            'BATCH_SIZE': conf.BATCH_SIZE,

            'VARIABLE_RANDOM_STANDARD_DEVIATION': conf.VARIABLE_RANDOM_STANDARD_DEVIATION,
            'BATCH_NORM_MEAN': conf.BATCH_NORM_MEAN,

            'TRAN_CONV_LAYER_3_HEIGHT': conf.TRAN_CONV_LAYER_3_HEIGHT,
            'TRAN_CONV_LAYER_3_WIDTH': conf.TRAN_CONV_LAYER_3_WIDTH,
            'TRAN_CONV_LAYER_2_HEIGHT': conf.TRAN_CONV_LAYER_2_HEIGHT,
            'TRAN_CONV_LAYER_2_WIDTH': conf.TRAN_CONV_LAYER_2_WIDTH,
            'TRAN_CONV_LAYER_1_HEIGHT': conf.TRAN_CONV_LAYER_1_HEIGHT,
            'TRAN_CONV_LAYER_1_WIDTH': conf.TRAN_CONV_LAYER_1_WIDTH,

            'FILTER_SIZE': conf.FILTER_SIZE,
            'CONV_STRIDE': conf.CONV_STRIDE,

            'TRAN_CONV_LAYER_1_OUT_CHANNEL': conf.TRAN_CONV_LAYER_1_OUT_CHANNEL,
            'TRAN_CONV_LAYER_2_OUT_CHANNEL': conf.TRAN_CONV_LAYER_2_OUT_CHANNEL,
            'TRAN_CONV_LAYER_3_OUT_CHANNEL': conf.TRAN_CONV_LAYER_3_OUT_CHANNEL,

        }
