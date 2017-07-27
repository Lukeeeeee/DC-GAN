import math

from src.config import Config


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class Step2VGGGeneratorConfig(Config):
    IN_HEIGHT = 28
    IN_WIDTH = 28
    IN_CHANNEL = 256

    OUT_HEIGHT = 224
    OUT_WIDTH = 224
    OUT_CHANNEL = 3

    TRAN_CONV_LAYER_1_IN_CHANNEL = 64
    TRAN_CONV_LAYER_2_IN_CHANNEL = 32
    TRAN_CONV_LAYER_3_IN_CHANNEL = 16

    FILTER_SIZE = 4

    CONV_STRIDE = 1

    TRAN_CONV_LAYER_3_HEIGHT = int(math.ceil(float(OUT_HEIGHT) / float(CONV_STRIDE)))
    TRAN_CONV_LAYER_3_WIDTH = int(math.ceil(float(OUT_WIDTH) / float(CONV_STRIDE)))

    TRAN_CONV_LAYER_2_HEIGHT = int(math.ceil(float(TRAN_CONV_LAYER_3_HEIGHT) / float(CONV_STRIDE)))
    TRAN_CONV_LAYER_2_WIDTH = int(math.ceil(float(TRAN_CONV_LAYER_3_WIDTH) / float(CONV_STRIDE)))

    TRAN_CONV_LAYER_1_HEIGHT = int(math.ceil(float(TRAN_CONV_LAYER_2_HEIGHT) / float(CONV_STRIDE)))
    TRAN_CONV_LAYER_1_WIDTH = int(math.ceil(float(TRAN_CONV_LAYER_2_WIDTH) / float(CONV_STRIDE)))

    G_LEARNING_RATE = 0.003

    BATCH_SIZE = 200

    VARIABLE_RANDOM_STANDARD_DEVIATION = 0.02

    BATCH_NORM_MEAN = 1.0

    BATCH_STANDARD_DEVIATION = 0.02

    PREFIX = 'Step2VGGGenerator_'

    @staticmethod
    def save_to_json(conf):
        return {
            conf.PREFIX + 'IN_HEIGHT': conf.IN_HEIGHT,
            conf.PREFIX + 'IN_WIDTH': conf.IN_WIDTH,
            conf.PREFIX + 'IN_CHANNEL': conf.IN_CHANNEL,

            conf.PREFIX + 'OUT_HEIGHT': conf.OUT_HEIGHT,
            conf.PREFIX + 'OUT_WIDTH': conf.OUT_WIDTH,
            conf.PREFIX + 'OUT_CHANNEL': conf.OUT_CHANNEL,

            conf.PREFIX + 'G_LEARNING_RATE': conf.G_LEARNING_RATE,

            conf.PREFIX + 'DATA_COUNT': conf.DATA_COUNT,
            conf.PREFIX + 'DATA_SOURCE': conf.DATA_SOURCE,
            conf.PREFIX + 'DATA_Z_NAME': conf.DATA_Z_NAME,

            conf.PREFIX + 'BATCH_SIZE': conf.BATCH_SIZE,

            conf.PREFIX + 'VARIABLE_RANDOM_STANDARD_DEVIATION': conf.VARIABLE_RANDOM_STANDARD_DEVIATION,
            conf.PREFIX + 'BATCH_NORM_MEAN': conf.BATCH_NORM_MEAN,

            conf.PREFIX + 'TRAN_CONV_LAYER_3_HEIGHT': conf.TRAN_CONV_LAYER_3_HEIGHT,
            conf.PREFIX + 'TRAN_CONV_LAYER_3_WIDTH': conf.TRAN_CONV_LAYER_3_WIDTH,
            conf.PREFIX + 'TRAN_CONV_LAYER_2_HEIGHT': conf.TRAN_CONV_LAYER_2_HEIGHT,
            conf.PREFIX + 'TRAN_CONV_LAYER_2_WIDTH': conf.TRAN_CONV_LAYER_2_WIDTH,
            conf.PREFIX + 'TRAN_CONV_LAYER_1_HEIGHT': conf.TRAN_CONV_LAYER_1_HEIGHT,
            conf.PREFIX + 'TRAN_CONV_LAYER_1_WIDTH': conf.TRAN_CONV_LAYER_1_WIDTH,

            conf.PREFIX + 'FILTER_SIZE': conf.FILTER_SIZE,
            conf.PREFIX + 'CONV_STRIDE': conf.CONV_STRIDE,

            conf.PREFIX + 'TRAN_CONV_LAYER_1_IN_CHANNEL': conf.TRAN_CONV_LAYER_1_IN_CHANNEL,
            conf.PREFIX + 'TRAN_CONV_LAYER_2_IN_CHANNEL': conf.TRAN_CONV_LAYER_2_IN_CHANNEL,
            conf.PREFIX + 'TRAN_CONV_LAYER_3_IN_CHANNEL': conf.TRAN_CONV_LAYER_3_IN_CHANNEL,

        }
