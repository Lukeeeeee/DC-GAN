import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import misc

from src.data.data import Data


class VGG16Data(Data):
    def __init__(self, data_path, config, model_file, load_image=True):
        super(VGG16Data, self).__init__(data_path=data_path, config=config)
        if load_image is True:
            self.image_set = self.load_data()
        else:
            self.image_set = None
        self.model_file = model_file

    def load_data(self):
        image_data = None
        for i in range(self.config.SAMPLE_COUNT):
            # data = misc.imread(name=self.data_path + 'new_cat.' + str(i) + '.jpg')
            d = '{0:06}'.format(i + 1)
            print(i)
            data = misc.imread(name=self.data_path + d + '.jpg')

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
        image_data = self.image_set[index * batch_size: (index + 1) * batch_size, ]
        image_data = np.reshape(np.ravel(image_data,
                                         order='C'),
                                newshape=[batch_size, self.config.DATA_WIDTH,
                                          self.config.DATA_HEIGHT, self.config.DATA_CHANNEL],
                                ).astype(np.float32)

        # TODO NORMAL THE PIC IS NECESSARY?
        # image_data = np.subtract(np.divide(image_data, 127.5), 1.0)
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

    def init_with_model(self, model_file):

        with open(model_file, mode='rb') as f:
            file_content = f.read()
        try:
            with tf.device('/gpu:1'):
                self.graph = tf.GraphDef()
                self.graph.ParseFromString(file_content)
                self.graph_input = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, self.config.DATA_WIDTH,
                                                         self.config.DATA_HEIGHT, self.config.DATA_CHANNEL])
                gpu_config = tf.GPUOptions(allow_growth=True)

                config = tf.ConfigProto(gpu_options=gpu_config, log_device_placement=True)

                self.sess = tf.Session(config=config)

                tf.import_graph_def(self.graph, input_map={'images': self.graph_input})
                self.graph = tf.get_default_graph()
                self.sess.run(tf.global_variables_initializer())
        except BaseException:
            self.graph = tf.GraphDef()
            self.graph.ParseFromString(file_content)
            self.graph_input = tf.placeholder(dtype=tf.float32,
                                              shape=[None, self.config.DATA_WIDTH,
                                                     self.config.DATA_HEIGHT, self.config.DATA_CHANNEL])
            gpu_config = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.4)

            config = tf.ConfigProto(gpu_options=gpu_config)

            self.sess = tf.Session(config=config)
            tf.import_graph_def(self.graph, input_map={'images': self.graph_input})
            self.graph = tf.get_default_graph()
            self.sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
    from dataset import DATASET_PATH

    data_path = '/home/mars/ANN/celeba/reshape_224_224/'
    model_file = DATASET_PATH + '/vgg16.tfmodel'
    from test.vggDeepGANTest.tempCelebaconfig.step2.step2VGGCelebaDataConfig import Step2VGGCelebaDataConfig

    a = VGG16Data(data_path=data_path, model_file=model_file, config=Step2VGGCelebaDataConfig())
    # a.init_with_model(model_file=model_file)
    for i in range(100):
        print(i)
        data = a.return_image_batch_data(batch_size=100, index=i)
        # res = a.eval_tensor_by_name(tensor_name=a.config.Z_SOURCE, image_batch=data)
        # res = np.reshape(res, newshape=[-1, 56, 56, 256])
        np.save(file=DATASET_PATH + '/celeba/224_224_3/step1_imagebatch_' + str(i) + '.npy', arr=data)
