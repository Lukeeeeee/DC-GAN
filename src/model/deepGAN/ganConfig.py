class GANConfig(object):
    TRAINING_EPOCH = 1000
    BATCH_SIZE = 200
    SAMPLE_COUNT = 10000
    BATCH_COUNT = int(SAMPLE_COUNT / BATCH_SIZE)

    @staticmethod
    def save_to_json(config):
        return {
            'TRAINING_EPOCH': config.TRAINING_EPOCH,
            'BATCH_SIZE': config.BATCH_SIZE
        }
