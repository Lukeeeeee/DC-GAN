import numpy as np
from PIL import Image
from scipy import misc

from dataset import DATASET_PATH
from src.data.data import Data


class VGG16Data(Data):
    def __init__(self, data_path, config):
        super(VGG16Data, self).__init__(data_path=data_path, config=config)
        self.image_set = self.load_data()

    def load_data(self):
        image_data = None
        for i in range(self.config.SAMPLE_COUNT):
            data = misc.imread(name=self.data_path + 'new_cat.' + str(i) + '.jpg')
            if i == 0:
                image_data = data
            else:
                image_data = np.concatenate((image_data, data))
        return image_data

    def return_z_batch_data(self, batch_size):
        z_batch = np.random.uniform(-1, 1, [batch_size, self.config.Z_WIDTH, self.config.Z_HEIGHT,
                                            self.config.Z_CHANNEL]).astype(np.float32)
        return z_batch

    def return_image_batch_data(self, batch_size, index):
        image_data = self.image_set[index: index + batch_size, ]
        image_data = np.reshape(np.ravel(image_data,
                                         order='C'),
                                newshape=[batch_size, self.config.IMAGE_WIDTH,
                                          self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNEL],
                                ).astype(np.float32)
        image_data = np.subtract(np.divide(image_data, 255), 0.5)
        return image_data

    @staticmethod
    def scale_image(data_path, pic_count, new_size):
        for i in range(pic_count):
            im = Image.open(data_path + 'new_cat.' + str(i) + '.jpg')
            im_new = im.resize(new_size, Image.ANTIALIAS)
            im_new.save(fp=data_path + 'new_cat.' + str(i) + '.jpg')
        pass


if __name__ == '__main__':
    path = DATASET_PATH + '/cat/'
    # VGG16Data.scale_image(data_path=path, pic_count=1000, new_size=(224, 224))
    a = VGG16Data(data_path=path, config=1)
