from src.config import Config


class Step2VGGDataConfig(Config):
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    IMAGE_CHANNEL = 3

    DATA_HEIGHT = 224
    DATA_WIDTH = 224
    DATA_CHANNEL = 3

    Z_HEIGHT = 28
    Z_WIDTH = 28
    Z_CHANNEL = 256

    SAMPLE_COUNT = 1000
    NPY_FILE_COUNT = 100

    Z_SOURCE = 'import/pool3'

    PREFIX = 'STEP2_VGG_DATA_'

    @staticmethod
    def save_to_json(config):
        return {
            config.PREFIX + 'IMAGE_HEIGHT': config.IMAGE_HEIGHT,
            config.PREFIX + 'IMAGE_WIDTH': config.IMAGE_WIDTH,
            config.PREFIX + 'IMAGE_CHANNEL': config.IMAGE_CHANNEL,
            config.PREFIX + 'Z_HEIGHT': config.Z_HEIGHT,
            config.PREFIX + 'Z_WIDTH': config.Z_WIDTH,
            config.PREFIX + 'Z_CHANNEL': config.Z_CHANNEL,
            config.PREFIX + 'Z_SOURCE': config.Z_SOURCE,
            config.PREFIX + 'SAMPLE_COUNT': config.SAMPLE_COUNT
        }
