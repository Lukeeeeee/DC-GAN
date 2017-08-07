import numpy as np

from src.data.mnist.mnistData import MnistConfig
from src.data.mnist.mnistData import MnistData


class Step3Data(MnistData):
    def __init__(self, data_path, config, mnist_cnn, sess):
        super(Step3Data, self).__init__(data_path=data_path, config=MnistConfig)
        self.image_set = self.load_data(count=10)
        self.mnist_cnn = mnist_cnn
        self.sess = sess
        self.new_config = config

    def return_z_batch_data(self, batch_size, index=None):
        # data = super(Step2Data, self).return_z_batch_data(batch_size)
        image = self.return_image_batch_data(batch_size=batch_size, index=index)
        data = self.mnist_cnn.eval_tensor(tensor=self.mnist_cnn.conv1,
                                          image_batch=image,
                                          keep_prob=0.5)
        data = np.reshape(data,
                          newshape=[-1, self.new_config.Z_WIDTH, self.new_config.Z_HEIGHT, self.new_config.Z_CHANNEL])
        return data
