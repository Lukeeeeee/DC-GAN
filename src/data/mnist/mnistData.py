import numpy as np
import scipy.io as sio
from PIL import Image

from dataset import DATASET_PATH
from mnistConfig import MnistConfig
from src.data.data import Data


class MnistData(Data):
    def __init__(self, data_path, config):
        super(MnistData, self).__init__(data_path=data_path, config=config)
        self.image_set = self.load_data()

    def load_data(self, count=None):
        if count is None:
            count = 1
        image_data = None
        for i in range(count):
            mat_file_path = self.data_path + '/digit' + str(i) + '.mat'
            data = sio.loadmat(mat_file_path)
            data = np.array(data['D'])
            if i == 0:
                image_data = data
            else:
                image_data = np.concatenate((image_data, data))
        return image_data

    def return_z_batch_data(self, batch_size, index=0):
        z_batch = np.random.uniform(-1, 1, [batch_size, self.config.Z_WIDTH, self.config.Z_HEIGHT,
                                            self.config.Z_CHANNEL]).astype(np.float32)
        return z_batch

    def return_image_batch_data(self, batch_size, index):
        image_data = self.image_set[index * batch_size: (index + 1) * batch_size, ]
        image_data = np.reshape(np.ravel(image_data,
                                         order='C'),
                                newshape=[batch_size, self.config.IMAGE_WIDTH,
                                          self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNEL],
                                ).astype(np.float32)
        # TODO NORMAL THE PIC IS NECESSARY?
        # image_data = np.subtract(np.divide(image_data, 255), 0.5)
        return image_data

    def show_pic(self, data):
        # im = Image.new(mode='L', size=(self.tempConfig.IMAGE_WIDTH, self.tempConfig.IMAGE_HEIGHT))
        data = np.reshape(data,
                          newshape=[self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT],
                          )
        data = np.multiply(np.add(data, 0.5), 255)
        im = Image.fromarray(data)
        im.show()


if __name__ == '__main__':
    config = MnistConfig()
    d = MnistData(data_path=DATASET_PATH + '/mnist', config=config)
    # print(d.return_image_batch_data(batch_size=200, index=0))
    data = d.return_image_batch_data(100, 1)
    d.show_pic(data=d.image_set[10:11, ])
