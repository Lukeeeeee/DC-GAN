from src.config import Config


class Step3GANConfig(Config):
    TRAINING_EPOCH = 100
    BATCH_SIZE = 100
    # SAMPLE_COUNT = 5000 0
    SAMPLE_COUNT = 5000  # 0 1
    BATCH_COUNT = int(SAMPLE_COUNT / BATCH_SIZE)
    # BATCH_COUNT = 1
    NAME = 'Step3_GAN'

    SAVE_MODEL_EVERY_EPOCH = 5

    TEST_EVERY_EPOCH = 1

    PREFIX = 'Step3GAN_'

    @staticmethod
    def save_to_json(config):
        return {
            config.PREFIX + 'TRAINING_EPOCH': config.TRAINING_EPOCH,
            config.PREFIX + 'BATCH_SIZE': config.BATCH_SIZE,
            config.PREFIX + 'BATCH_COUNT': config.BATCH_COUNT,
            config.PREFIX + 'SAMPLE_COUNT': config.SAMPLE_COUNT,
            config.PREFIX + 'NAME': config.NAME
        }


if __name__ == '__main__':
    a = Step3GANConfig()
    a.log_config('1.json')
