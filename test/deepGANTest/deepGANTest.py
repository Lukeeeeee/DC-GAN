import tensorflow as tf

from dataset import DATASET_PATH
from src.data.deepGAN.step1Data import Step1Data
from src.data.deepGAN.step1DataConfig import Step1DataConfig
from src.data.deepGAN.step2Data import Step2Data
from src.data.deepGAN.step2DataConfig import Step2DataConfig
from src.model.basicGAN.basicGAN import BasicGAN
from test.deepGANTest.tempConfig.step1 import *
from test.deepGANTest.tempConfig.step2 import *

if __name__ == '__main__':
    # Init first GAN

    data_sess = tf.InteractiveSession()
    config = Step1DataConfig()
    step1_data = Step1Data(data_path=DATASET_PATH + '/mnist',
                           config=config,
                           sess=data_sess,
                           mnist_cnn=None)

    step1_sess = tf.InteractiveSession()
    gan_config = Step1GANConfig()
    d_config = Step1DiscriminatorConfig()
    g_config = Step1GeneratorConfig()
    step1_gan = BasicGAN(config=gan_config,
                         sess=step1_sess,
                         data=step1_data,
                         g_config=g_config,
                         d_config=d_config)

    # Init second GAN

    config = Step2DataConfig()
    step2_data = Step2Data(data_path=DATASET_PATH + '/mnist',
                           config=config,
                           sess=data_sess,
                           mnist_cnn=None)

    step2_sess = tf.InteractiveSession()
    gan_config = Step2GANConfig()
    d_config = Step2DiscriminatorConfig()
    g_config = Step2GeneratorConfig()
    step2_gan = BasicGAN(config=gan_config,
                         sess=step2_sess,
                         data=step2_data,
                         g_config=g_config,
                         d_config=d_config)
