from src.data.data import Data


class Step2Data(Data):
    def __init__(self, data_path, config, mnist_cnn, sess):
        super(Step2Data, self).__init__(data_path=data_path, config=config)
        self.mnist_cnn = mnist_cnn
        self.sess = sess

    def return_image_batch_data(self, batch_size, index):
        return None
