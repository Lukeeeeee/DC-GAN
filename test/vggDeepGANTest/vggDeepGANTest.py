import tensorflow as tf

from dataset import DATASET_PATH
from src.data.deepVGG16Data import *
from src.model.deepGAN.deepGAN import DeepGAN
from test.vggDeepGANTest.tempConfig import *

if __name__ == '__main__':
    data_path = DATASET_PATH + '/cat/'
    model_file = DATASET_PATH + '/vgg16.tfmodel'
    step1_config = Step1VGGDataConfig()
    step1_data = Step1VGGData(config=step1_config, data_path=data_path, model_file=model_file)

    gan_config = Step1VGGGANConfig()
    g_config = Step1VGGGeneratorConfig()
    d_config = Step1VGGDiscriminatorConfig()
    step1_gan = DeepGAN(config=gan_config,
                        sess=tf.InteractiveSession(),
                        data=step1_data,
                        g_config=g_config,
                        d_config=d_config)

    step2_config = Step2VGGDataConfig()
    step2_data = Step2VGGData(config=step2_config, data_path=data_path, model_file=model_file)

    gan_config = Step2VGGGANConfig()
    g_config = Step2VGGGeneratorConfig()
    d_config = Step2VGGDiscriminatorConfig()

    step2_gan = DeepGAN(config=gan_config,
                        sess=tf.InteractiveSession(),
                        data=step2_data,
                        g_config=g_config,
                        d_config=d_config)
    step1_gan.train()
    step2_gan.train()
