"""
TEST 1.0
Usage:
    vggCelbaDeepGANTest.py train
    vggCelbaDeepGANTest.py test model_id <model_id> epoch <epoch>
"""

import os
import sys
from docopt import docopt

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.join(ROOT_PATH, '../../')

sys.path.append(ROOT_PATH)
sys.path.append(CURRENT_PATH)

from tempCelebaconfig import *
from dataset import DATASET_PATH
from src.data.deepVGG16Data import *
import tensorflow as tf
from src.model.deepGAN.deepGAN import DeepGAN
from log import LOG_PATH
import numpy as np
from PIL import Image


def train_step2():
    data_path = DATASET_PATH + '/celeba/224_224_3/'
    z_path = DATASET_PATH + '/celeba/56_56_256/'
    model_file = DATASET_PATH + '/vgg16.tfmodel'

    step2_config = Step2VGGCelebaDataConfig()
    step2_data = Step2VGGData(config=step2_config,
                              image_data_path=data_path,
                              model_file=model_file,
                              z_data_path=z_path,
                              load_image=False)

    gan_config = Step2VGGCelbaeGANConfig()
    g_config = Step2VGGCelebaGeneratorConfig()
    d_config = Step2VGGCelebaDiscriminatorConfig()

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

    pass


def load_test_step2(model_path, epoch):
    data_path = DATASET_PATH + '/celeba/224_224_3/'
    z_path = DATASET_PATH + '/celeba/56_56_256/'
    model_file = DATASET_PATH + '/vgg16.tfmodel'

    step2_config = Step2VGGCelebaDataConfig()
    step2_data = Step2VGGData(config=step2_config,
                              image_data_path=data_path,
                              model_file=model_file,
                              z_data_path=z_path,
                              load_image=False)

    gan_config = Step2VGGCelbaeGANConfig()
    g_config = Step2VGGCelebaGeneratorConfig()
    d_config = Step2VGGCelebaDiscriminatorConfig()

    gpu_config = tf.GPUOptions(allow_growth=True)

    config = tf.ConfigProto(gpu_options=gpu_config, log_device_placement=True)

    sess = tf.Session(config=config)

    step2_gan = DeepGAN(config=gan_config,
                        sess=sess,
                        data=step2_data,
                        g_config=g_config,
                        d_config=d_config,
                        step2_flag=True)

    step2_gan.load_model(model_path=model_path, epoch=epoch)
    image, z = step2_gan.data.return_batch_data(20, 0)
    res = step2_gan.eval_tensor(tensor=step2_gan.G.output, image_batch=image, z_batch=z)
    for i in range(5):
        data = np.multiply(np.add(res, 1.0), 127.5)
        data = np.reshape(data[i:i + 1, ],
                          newshape=[224, 224, 3],
                          ).astype((np.uint8))
        im = Image.fromarray(data, mode='RGB')
        im.show()
    for i in range(5):
        # data = np.multiply(np.add(image, 1.0), 127.5)
        data = np.reshape(image[i:i + 1, ],
                          newshape=[224, 224, 3],
                          ).astype((np.uint8))
        im = Image.fromarray(data, mode='RGB')
        im.show()


if __name__ == '__main__':

    arguments = docopt(__doc__)
    if arguments['train']:
        train_step2()
    else:
        path = LOG_PATH + '/' + arguments['<model_id>'] + '/Step2_VGG_Celeba_GAN/model/'
        load_test_step2(path, arguments['<epoch>'])
