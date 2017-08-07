import numpy as np

from src.data.mnist.mnistData import MnistConfig
from src.data.mnist.mnistData import MnistData


class Step1Data(MnistData):
    def __init__(self, data_path, config, mnist_cnn, sess):
        super(Step1Data, self).__init__(data_path=data_path, config=MnistConfig)
        self.image_set = self.load_data(count=10)
        self.mnist_cnn = mnist_cnn
        self.sess = sess
        self.new_config = config
        # mnist_config = MnistCNNDataConfig()
        # self.mnist_data = MnistCNNData(data_path=DATASET_PATH + '/mnist/',
        #                                config=mnist_config)

    def return_image_batch_data(self, batch_size, index):
        image = super(Step1Data, self).return_image_batch_data(batch_size, index)
        data = self.mnist_cnn.eval_tensor(tensor=self.mnist_cnn.conv1,
                                          image_batch=image,
                                          keep_prob=0.5,
                                          )
        data = np.reshape(data,
                          newshape=[-1, self.new_config.IMAGE_WIDTH, self.new_config.IMAGE_HEIGHT,
                                    self.new_config.IMAGE_CHANNEL])

        return data
