import math

from src.config import Config


class Step3DiscriminatorConfig(Config):
    IN_HEIGHT = 28
    IN_WIDTH = 28
    IN_CHANNEL = 1

    LEARNING_RATE = 0.01

    CONV_LAYER_COUNT = 3

    CONV_LAYER_1_OUT_CHANNEL = 6

    CONV_LAYER_2_OUT_CHANNEL = 12

    CONV_LAYER_3_OUT_CHANNEL = 24

    CONV_STRIDE = 2

    FILTER_SIZE = 5

    VARIABLE_RANDOM_STANDARD_DEVIATION = 0.02

    BATCH_NORM_MEAN = 0.0

    BATCH_STANDARD_DEVIATION = 0.02

    CONV_OUT_HEIGHT = int(math.ceil(float(IN_HEIGHT) / float(math.pow(CONV_STRIDE, CONV_LAYER_COUNT))))
    CONV_OUT_WIDTH = int(math.ceil(float(IN_WIDTH) / float(math.pow(CONV_STRIDE, CONV_LAYER_COUNT))))

    OUTPUT_SIZE = 1

    PREFIX = 'Step3Discriminator_'

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
    a = Step3DiscriminatorConfig()
    a.log_config('1.json')
