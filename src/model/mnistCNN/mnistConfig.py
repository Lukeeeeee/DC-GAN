import math

from src.config import Config


class MnistCNNConfig(Config):
    IN_HEIGHT = 28
    IN_WIDTH = 28
    IN_CHANNEL = 1

    LEARNING_RATE = 0.001
    EPOCH = 100
    SAMPLE_COUNT = 60000
    BATCH_SIZE = 200
    BATCH_COUNT = SAMPLE_COUNT // BATCH_SIZE

    CONV_LAYER_COUNT = 3

    CONV_LAYER_1_OUT_CHANNEL = 4

    CONV_LAYER_2_OUT_CHANNEL = 16

    CONV_LAYER_3_OUT_CHANNEL = 32

    CONV_STRIDE = 1

    FILTER_SIZE = 5
    MAX_POOL = 2

    CONV_OUT_HEIGHT = int(math.ceil(float(IN_HEIGHT) / float(math.pow(MAX_POOL, CONV_LAYER_COUNT))))
    CONV_OUT_WIDTH = int(math.ceil(float(IN_WIDTH) / float(math.pow(MAX_POOL, CONV_LAYER_COUNT))))

    FULLY_CONNECTED_IN = CONV_OUT_WIDTH * CONV_OUT_HEIGHT * CONV_LAYER_3_OUT_CHANNEL

    FULLY_CONNECTED_OUT = 1024

    OUTPUT_SIZE = 10

    PREFIX = 'MNIST_'

    @staticmethod
    def save_to_json(config):
        return {

        }


if __name__ == '__main__':
    a = MnistCNNConfig()
