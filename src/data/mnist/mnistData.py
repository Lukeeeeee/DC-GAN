import numpy as np
import scipy.io as sio
from PIL import Image

from dataset import DATASET_PATH
from mnistConfig import MnistConfig as conf
from src.data.data import Data


class MnistData(Data):
    def __init__(self, data_path, data_config=None):
        super(MnistData, self).__init__(data_path=data_path, data_config=data_config)
        self.image_set = self.load_data()

    def load_data(self):
        image_data = None
        for i in range(10):
            mat_file_path = self.data_path + '/digit' + str(i) + '.mat'
            data = sio.loadmat(mat_file_path)
            data = np.array(data['D'])
            if i == 0:
                image_data = data
            else:
                image_data = np.concatenate((image_data, data))
        return image_data

    def return_z_batch_data(self, batch_size):
        z_batch = np.random.uniform(-1, 1, [batch_size, conf.Z_WIDTH, conf.Z_HEIGHT, conf.Z_CHANNEL]).astype(np.float32)
        return z_batch

    def return_image_batch_data(self, batch_size, index):
        image_data = self.image_set[index: index + batch_size, ]
        image_data = np.reshape(image_data,
                                newshape=[batch_size, conf.IMAGE_WIDTH,
                                          conf.IMAGE_HEIGHT, conf.IMAGE_CHANNEL]).astype(np.float32)
        # image_data = np.divide(image_data, 255)
        return image_data

    def show_pic(self, data):
        # im = Image.new(mode='L', size=(conf.IMAGE_WIDTH, conf.IMAGE_HEIGHT))
        data = np.reshape(data, newshape=[conf.IMAGE_WIDTH, conf.IMAGE_HEIGHT])
        im = Image.fromarray(data, 'L')
        im.show()


if __name__ == '__main__':
    d = MnistData(data_path=DATASET_PATH + '/mnist')
    # print(d.return_image_batch_data(batch_size=200, index=0))
    i = 40000
    d.show_pic(data=d.image_set[i:i + 1, ])
