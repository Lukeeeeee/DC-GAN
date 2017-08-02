import math

from src.config import Config


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class Step2VGGCelebaGeneratorConfig(Config):
    IN_HEIGHT = 56
    IN_WIDTH = 56
    IN_CHANNEL = 256

    TRAN_CONV_LAYER_1_IN_CHANNEL = 128
    TRAN_CONV_LAYER_2_IN_CHANNEL = 64
    TRAN_CONV_LAYER_3_IN_CHANNEL = 64
    TRAN_CONV_LAYER_4_IN_CHANNEL = 32

    OUT_HEIGHT = 224
    OUT_WIDTH = 224
    OUT_CHANNEL = 3

    FILTER_SIZE = 2

    CONV_1_STRIDE = 1
    CONV_2_STRIDE = 2
    CONV_3_STRIDE = 1
    CONV_4_STRIDE = 2
    CONV_5_STRIDE = 1

    TRAN_CONV_LAYER_5_HEIGHT = OUT_HEIGHT
    TRAN_CONV_LAYER_5_WIDTH = OUT_WIDTH

    TRAN_CONV_LAYER_4_HEIGHT = int(math.ceil(float(TRAN_CONV_LAYER_5_HEIGHT) / float(CONV_5_STRIDE)))
    TRAN_CONV_LAYER_4_WIDTH = int(math.ceil(float(TRAN_CONV_LAYER_5_WIDTH) / float(CONV_5_STRIDE)))

    TRAN_CONV_LAYER_3_HEIGHT = int(math.ceil(float(TRAN_CONV_LAYER_4_HEIGHT) / float(CONV_4_STRIDE)))
    TRAN_CONV_LAYER_3_WIDTH = int(math.ceil(float(TRAN_CONV_LAYER_4_WIDTH) / float(CONV_4_STRIDE)))

    TRAN_CONV_LAYER_2_HEIGHT = int(math.ceil(float(TRAN_CONV_LAYER_3_HEIGHT) / float(CONV_3_STRIDE)))
    TRAN_CONV_LAYER_2_WIDTH = int(math.ceil(float(TRAN_CONV_LAYER_3_WIDTH) / float(CONV_3_STRIDE)))

    TRAN_CONV_LAYER_1_HEIGHT = int(math.ceil(float(TRAN_CONV_LAYER_2_HEIGHT) / float(CONV_2_STRIDE)))
    TRAN_CONV_LAYER_1_WIDTH = int(math.ceil(float(TRAN_CONV_LAYER_2_WIDTH) / float(CONV_2_STRIDE)))

    assert TRAN_CONV_LAYER_1_HEIGHT == IN_HEIGHT
    assert TRAN_CONV_LAYER_1_WIDTH == IN_WIDTH

    G_LEARNING_RATE = 0.03

    BATCH_SIZE = 20

    VARIABLE_RANDOM_STANDARD_DEVIATION = 0.02

    BATCH_NORM_MEAN = 1.0

    BATCH_STANDARD_DEVIATION = 0.02

    PREFIX = 'Step2VGGCelebaGenerator_'

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


if __name__ == '__main__':
    a = Step2VGGCelebaGeneratorConfig()
