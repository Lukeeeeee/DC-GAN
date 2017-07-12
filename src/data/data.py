import numpy as np


class Data(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.sample_count = None
        pass

    def return_image_batch_data(self, batch_size, index):
        return None

    def return_z_batch_data(self, batch_size):
        z_batch = np.random.uniform(-1, 1, [batch_size, 1, 1, 100]).astype(np.float32)
        return z_batch

    def return_batch_data(self, batch_size, index):
        image = self.return_image_batch_data(batch_size, index)
        z = self.return_z_batch_data(batch_size)
        return image, z


if __name__ == '__main__':
    a = Data(data_path=1)
    a.return_z_batch_data(100)
