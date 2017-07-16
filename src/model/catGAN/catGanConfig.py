from src.config import Config


class CatGANConfig(Config):
    TRAINING_EPOCH = 10000000
    BATCH_SIZE = 200
    SAMPLE_COUNT = 50000
    BATCH_COUNT = int(SAMPLE_COUNT / BATCH_SIZE)
    # BATCH_COUNT = 1

    SAVE_MODEL_EVERY_EPOCH = 1

    PREFIX = 'GAN_'

    @staticmethod
    def save_to_json(config):
        return {
            config.PREFIX + 'TRAINING_EPOCH': config.TRAINING_EPOCH,
            config.PREFIX + 'BATCH_SIZE': config.BATCH_SIZE,
            config.PREFIX + 'BATCH_COUNT': config.BATCH_COUNT
        }


if __name__ == '__main__':
    a = CatGANConfig()
    a.log_config('1.json')
