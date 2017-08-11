from src.config import Config


class Step3DataConfig(Config):
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    IMAGE_CHANNEL = 1

    Z_HEIGHT = 7
    Z_WIDTH = 7
    Z_CHANNEL = 16

    PREFIX = 'STEP3DATA_'

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
