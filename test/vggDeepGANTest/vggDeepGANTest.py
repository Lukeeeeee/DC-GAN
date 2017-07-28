import tensorflow as tf

from dataset import DATASET_PATH
from src.data.deepVGG16Data import *
from src.model.deepGAN.deepGAN import DeepGAN
from test.vggDeepGANTest.tempConfig import *


def train_step1():
    data_path = DATASET_PATH + '/cat/'
    model_file = DATASET_PATH + '/vgg16.tfmodel'

    # step1_config = Step1VGGDataConfig()
    # step1_data = Step1VGGData(config=step1_config, data_path=data_path, model_file=model_file)

    # gan_config = Step1VGGGANConfig()
    # g_config = Step1VGGGeneratorConfig()
    # d_config = Step1VGGDiscriminatorConfig()
    # step1_gan = DeepGAN(config=gan_config,
    #                     sess=tf.InteractiveSession(),
    #                     data=step1_data,
    #                     g_config=g_config,
    #                     d_config=d_config)
    # step1_gan.train()


def train_step2():
    data_path = DATASET_PATH + '/cat/'
    model_file = DATASET_PATH + '/vgg16.tfmodel'

    step2_config = Step2VGGDataConfig()
    step2_data = Step2VGGData(config=step2_config, data_path=data_path, model_file=model_file)

    gan_config = Step2VGGGANConfig()
    g_config = Step2VGGGeneratorConfig()
    d_config = Step2VGGDiscriminatorConfig()

    gpu_config = tf.GPUOptions(allow_growth=True)

    config = tf.ConfigProto(gpu_options=gpu_config, log_device_placement=True)

    sess = tf.Session(config=config)

    step2_gan = DeepGAN(config=gan_config,
                        sess=sess,
                        data=step2_data,
                        g_config=g_config,
                        d_config=d_config,
                        step2_flag=True)
    step2_gan.train()


def load_step2_test():
    data_path = DATASET_PATH + '/cat/'
    model_file = DATASET_PATH + '/vgg16.tfmodel'

    step2_config = Step2VGGDataConfig()
    step2_data = Step2VGGData(config=step2_config, data_path=data_path, model_file=model_file)

    gan_config = Step2VGGGANConfig()
    g_config = Step2VGGGeneratorConfig()
    d_config = Step2VGGDiscriminatorConfig()

    gpu_config = tf.GPUOptions(allow_growth=True)

    config = tf.ConfigProto(gpu_options=gpu_config, log_device_placement=True)

    sess = tf.Session(config=config)

    step2_gan = DeepGAN(config=gan_config,
                        sess=sess,
                        data=step2_data,
                        g_config=g_config,
                        d_config=d_config,
                        step2_flag=True)
    from log import LOG_PATH
    import numpy as np
    from PIL import Image
    step2_gan.load_model(model_path=LOG_PATH + '/7-28-11-18-7/Step2_VGG_GAN/model/', epoch=100)
    image, z = step2_gan.data.return_batch_data(50, 0)
    res = step2_gan.eval_tensor(tensor=step2_gan.G.output, image_batch=image, z_batch=z)
    for i in range(10):
        data = np.multiply(np.add(res, 0.5), 255)
        data = np.reshape(data[i:i + 1, ],
                          newshape=[224, 224, 3],
                          ).astype((np.uint8))
        im = Image.fromarray(data, mode='RGB')
        im.show()

if __name__ == '__main__':
    load_step2_test()
