from src.config import Config


class MnistCNNDataConfig(Config):
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    IMAGE_CHANNEL = 1

    PREFIX = 'MNISTCNN_'

    @staticmethod
    def save_to_json(config):
        return {
            config.PREFIX + 'IMAGE_HEIGHT': config.IMAGE_HEIGHT,
            config.PREFIX + 'IMAGE_WIDTH': config.IMAGE_WIDTH,
            config.PREFIX + 'IMAGE_CHANNEL': config.IMAGE_CHANNEL,

        }


if __name__ == '__main__':
    a = MnistCNNDataConfig()
    a.log_config('1.json')
