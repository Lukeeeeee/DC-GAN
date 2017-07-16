import numpy as np


class Data(object):
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.IMAGE_HEIGHT = None
        self.IMAGE_WIDTH = None
        self.IMAGE_CHANNEL = None

        self.Z_HEIGHT = None
        self.Z_WIDTH = None
        self.Z_CHANNEL = None
        self.config = config

    def return_image_batch_data(self, batch_size, index):
        return None

    def return_z_batch_data(self, batch_size):
        z_batch = np.random.uniform(-1, 1, [batch_size, 1, 1, 100]).astype(np.float32)
        return z_batch

    def return_batch_data(self, batch_size, index):
        image = self.return_image_batch_data(batch_size, index)
        z = self.return_z_batch_data(batch_size)
        return image, z

    def log_config(self, log_file):
        self.config.log_config(self.config, log_file)

if __name__ == '__main__':
    a = Data(data_path=1, )
    a.return_z_batch_data(100)
