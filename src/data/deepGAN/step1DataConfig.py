from src.config import Config


class Step1DataConfig(Config):
    IMAGE_HEIGHT = 7
    IMAGE_WIDTH = 7
    IMAGE_CHANNEL = 16

    Z_HEIGHT = 1
    Z_WIDTH = 1
    Z_CHANNEL = 100

    PREFIX = 'STEP1DATA_'

    @staticmethod
    def save_to_json(config):
        return {
            config.PREFIX + 'IMAGE_HEIGHT': config.IMAGE_HEIGHT,
            config.PREFIX + 'IMAGE_WIDTH': config.IMAGE_WIDTH,
            config.PREFIX + 'IMAGE_CHANNEL': config.IMAGE_CHANNEL,
            config.PREFIX + 'Z_HEIGHT': config.Z_HEIGHT,
            config.PREFIX + 'Z_WIDTH': config.Z_WIDTH,
            config.PREFIX + 'Z_CHANNEL': config.Z_CHANNEL

        }
