import numpy as np
import scipy.io as sio

from src.data.mnist.mnistData import MnistData


class MnistCNNData(MnistData):
    def __init__(self, data_path, config):
        super(MnistCNNData, self).__init__(data_path=data_path, config=config)

        # self.image_set = self.load_data(count=10)
        image_data = None
        for i in range(10):
            mat_file_path = self.data_path + '/digit' + str(i) + '.mat'
            data = sio.loadmat(mat_file_path)
            data = np.array(data['D'])
            size = data.shape[0]
            label = np.full([size], i * 1.0)
            data = np.insert(data, 0, label, axis=1)
            if i == 0:
                image_data = data
            else:
                image_data = np.concatenate((image_data, data))

        self.image_set = image_data

        np.random.shuffle(self.image_set)

    def return_image_batch_data(self, batch_size, index):
        image_data = self.image_set[index: index + batch_size, 1:]
        image_data = np.reshape(np.ravel(image_data,
                                         order='C'),
                                newshape=[batch_size, self.config.IMAGE_WIDTH,
                                          self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNEL],
                                ).astype(np.float32)
        image_data = np.subtract(np.divide(image_data, 255), 0.5)
        return image_data

    def return_batch_data(self, batch_size, index):
        lable_data = self.image_set[index: index + batch_size, 0:1]
        lable_data = np.reshape(lable_data, newshape=[batch_size]).astype(np.int32)
        image_data = self.return_image_batch_data(batch_size, index)
        return image_data, lable_data


if __name__ == '__main__':
    from dataset import DATASET_PATH
    from src.data.mnistCNN.mnistCNNDataConfig import MnistCNNDataConfig

    config = MnistCNNDataConfig()
    a = MnistCNNData(DATASET_PATH + '/mnist', config=config)
