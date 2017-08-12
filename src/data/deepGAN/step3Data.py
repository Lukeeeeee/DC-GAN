import numpy as np
from dataset import DATASET_PATH
from src.data.mnist.mnistData import MnistConfig
from src.data.mnist.mnistData import MnistData


class Step3Data(MnistData):
    def __init__(self, data_path, config, mnist_cnn=None, sess=None):
        super(Step3Data, self).__init__(data_path=data_path, config=MnistConfig)
        self.image_set = self.load_data(count=10)
        self.z_set = self.load_z_data(data_path=DATASET_PATH + '/mnist/mnist_7_7_16/')
        self.mnist_cnn = mnist_cnn
        self.sess = sess
        self.new_config = config

    def load_z_data(self, data_path):
        z_data = None
        for i in range(10):
            res = np.load(data_path + str(i) + '.npy')
            if i == 0:
                z_data = res
            else:
                z_data = np.concatenate((z_data, res))
        return z_data

    def return_z_batch_data(self, batch_size, index=None):
        # data = super(Step2Data, self).return_z_batch_data(batch_size)
        z = self.z_set[index * batch_size: (index + 1) * batch_size, ]
        data = np.reshape(z,
                          newshape=[-1, self.new_config.Z_WIDTH, self.new_config.Z_HEIGHT, self.new_config.Z_CHANNEL])
        return data
