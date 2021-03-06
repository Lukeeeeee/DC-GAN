import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import misc

from src.data.data import Data


class VGG16Data(Data):
    def __init__(self, data_path, config, model_file):
        super(VGG16Data, self).__init__(data_path=data_path, config=config)
        self.image_set = self.load_data()
        with open(model_file, mode='rb') as f:
            file_content = f.read()
        self.graph = tf.GraphDef()
        self.graph.ParseFromString(file_content)
        self.graph_input = tf.placeholder(dtype=tf.float32,
                                          shape=[None, self.config.DATA_WIDTH,
                                                 self.config.DATA_HEIGHT, self.config.DATA_CHANNEL])
        self.sess = tf.InteractiveSession()
        tf.import_graph_def(self.graph, input_map={'images': self.graph_input})
        self.graph = tf.get_default_graph()
        self.sess.run(tf.global_variables_initializer())

    def load_data(self):
        image_data = None
        for i in range(self.config.SAMPLE_COUNT):
            data = misc.imread(name=self.data_path + 'new_cat.' + str(i) + '.jpg')
            data = np.reshape(np.array(data),
                              newshape=[1, self.config.DATA_WIDTH,
                                        self.config.DATA_HEIGHT, self.config.DATA_CHANNEL])
            if i == 0:
                image_data = data
            else:
                image_data = np.concatenate((image_data, data))
        return image_data

    def return_z_batch_data(self, batch_size, index=None):
        z_batch = np.random.uniform(-1, 1, [batch_size, self.config.Z_WIDTH, self.config.Z_HEIGHT,
                                            self.config.Z_CHANNEL]).astype(np.float32)
        return z_batch

    def return_image_batch_data(self, batch_size, index):
        image_data = self.image_set[index: index + batch_size, ]
        image_data = np.reshape(np.ravel(image_data,
                                         order='C'),
                                newshape=[batch_size, self.config.DATA_WIDTH,
                                          self.config.DATA_HEIGHT, self.config.DATA_CHANNEL],
                                ).astype(np.float32)

        # TODO NORMAL THE PIC IS NECESSARY?
        # image_data = np.subtract(np.divide(image_data, 255), 0.5)
        return image_data

    def eval_tensor_by_name(self, tensor_name, image_batch):
        try:
            tensor = self.graph.get_tensor_by_name(tensor_name)
        except BaseException:
            operation = self.graph.get_operation_by_name(tensor_name)
            tensor = operation.outputs
        res = self.sess.run(fetches=tensor,
                            feed_dict={self.graph_input: image_batch})
        return res

    @staticmethod
    def scale_image(data_path, pic_count, new_size):
        for i in range(pic_count):
            im = Image.open(data_path + 'new_cat.' + str(i) + '.jpg')
            im_new = im.resize(new_size, Image.ANTIALIAS)
            im_new.save(fp=data_path + 'new_cat.' + str(i) + '.jpg')
        pass
