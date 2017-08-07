import math

from src.config import Config


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class Step3GeneratorConfig(Config):
    IN_HEIGHT = 14
    IN_WIDTH = 14
    IN_CHANNEL = 4

    OUT_HEIGHT = 28
    OUT_WIDTH = 28
    OUT_CHANNEL = 1

    FILTER_SIZE = 4

    CONV_STRIDE = 2

    G_LEARNING_RATE = 0.003

    BATCH_SIZE = 200

    VARIABLE_RANDOM_STANDARD_DEVIATION = 0.02

    BATCH_NORM_MEAN = 1.0

    BATCH_STANDARD_DEVIATION = 0.02

    PREFIX = 'Step3Generator_'

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

            conf.PREFIX + 'BATCH_SIZE': conf.BATCH_SIZE,

            conf.PREFIX + 'VARIABLE_RANDOM_STANDARD_DEVIATION': conf.VARIABLE_RANDOM_STANDARD_DEVIATION,
            conf.PREFIX + 'BATCH_NORM_MEAN': conf.BATCH_NORM_MEAN,

            conf.PREFIX + 'FILTER_SIZE': conf.FILTER_SIZE,
            conf.PREFIX + 'CONV_STRIDE': conf.CONV_STRIDE,

        }


if __name__ == '__main__':
    a = Step3GeneratorConfig()
    a.log_config('1.json')
