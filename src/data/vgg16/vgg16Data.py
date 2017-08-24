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
        for i in range(self.config.NPY_FILE_COUNT):
            data = np.load(file=self.data_path + 'step1_imagebatch_' + str(i) + '.npy')
            data = np.reshape(np.array(data),
                              newshape=[-1, self.config.DATA_WIDTH,
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

    @staticmethod
    def print_all_tensor():
        tensor = tf.get_default_graph().get_operations()
        for node in tensor:
            try:
                print(node.name, node.outputs)
            except BaseException:
                print(node.name)

if __name__ == '__main__':
    from dataset import DATASET_PATH

    data_path = DATASET_PATH + '/celeba/224_224_3/'
    model_file = DATASET_PATH + '/vgg16.tfmodel'
    from test.vggDeepGANTest.tempCelebaconfig.step2.step2VGGCelebaDataConfig import Step2VGGCelebaDataConfig

    a = VGG16Data(data_path=data_path, model_file=model_file, config=Step2VGGCelebaDataConfig(), load_image=True)
    for i in range(Step2VGGCelebaDataConfig().NPY_FILE_COUNT):
        print(i)
        data = a.return_image_batch_data(batch_size=100, index=i)
        res = a.eval_tensor_by_name(tensor_name='import/pool1', image_batch=data)
        res = np.reshape(res, newshape=[-1, 112, 112, 64])
        np.save(file=DATASET_PATH + '/celeba/112_112_64/step1_imagebatch_' + str(i) + '.npy', arr=res)

        res = a.eval_tensor_by_name(tensor_name='import/pool2', image_batch=data)
        res = np.reshape(res, newshape=[-1, 56, 56, 128])
        np.save(file=DATASET_PATH + '/celeba/56_56_128/step1_imagebatch_' + str(i) + '.npy', arr=res)

        res = a.eval_tensor_by_name(tensor_name='import/pool3', image_batch=data)
        res = np.reshape(res, newshape=[-1, 28, 28, 256])
        np.save(file=DATASET_PATH + '/celeba/28_28_256/step1_imagebatch_' + str(i) + '.npy', arr=res)

        res = a.eval_tensor_by_name(tensor_name='import/pool4', image_batch=data)
        res = np.reshape(res, newshape=[-1, 14, 14, 512])
        np.save(file=DATASET_PATH + '/celeba/14_14_512/step1_imagebatch_' + str(i) + '.npy', arr=res)

        res = a.eval_tensor_by_name(tensor_name='import/pool5', image_batch=data)
        res = np.reshape(res, newshape=[-1, 7, 7, 512])
        np.save(file=DATASET_PATH + '/celeba/7_7_512/step1_imagebatch_' + str(i) + '.npy', arr=res)
