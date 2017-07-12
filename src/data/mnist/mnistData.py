import numpy as np
import scipy.io as sio

from dataset import DATASET_PATH
from src.data.data import Data


class MnistData(Data):

    def __init__(self, data_path):
        super(MnistData, self).__init__(data_path=data_path)
        self.image_set = self.load_data()

        self.HEIGHT = 28
        self.WIDTH = 28

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

    def return_image_batch_data(self, batch_size, index):
        image_data = self.image_set[index: index + batch_size, ]
        image_data = np.reshape(image_data, newshape=[batch_size, self.WIDTH, self.HEIGHT, 1]).astype(np.float32)
        return image_data


if __name__ == '__main__':
    d = MnistData(data_path=DATASET_PATH + '/mnist')
    print(d.return_image_batch_data(batch_size=200, index=0))
