from dataset import DATASET_PATH
from src.data.deepVGG16Data import *

if __name__ == '__main__':
    data_path = DATASET_PATH + '/cat/'
    model_file = DATASET_PATH + '/vgg16.tfmodel'

    step1_config = Step1VGGDataConfig()
    step1_data = Step1VGGData(config=step1_config, data_path=data_path, model_file=model_file)


    # gan_config = Step1VGGGANConfig()
    # g_config = Step1VGGGeneratorConfig()
    # d_config = Step1VGGDiscriminatorConfig()
    # step1_gan = DeepGAN(config=gan_config,
    #                     sess=tf.InteractiveSession(),
    #                     data=step1_data,
    #                     g_config=g_config,
    #                     d_config=d_config)
    # step1_gan.train()

    # step2_config = Step2VGGDataConfig()
    # step2_data = Step2VGGData(config=step2_config, data_path=data_path, model_file=model_file)


    # gan_config = Step2VGGGANConfig()
    # g_config = Step2VGGGeneratorConfig()
    # d_config = Step2VGGDiscriminatorConfig()
    #
    # gpu_config = tf.GPUOptions(allow_growth=True)
    #
    # config = tf.ConfigProto(gpu_options=gpu_config, log_device_placement=True)
    #
    # sess = tf.Session(config=config)
    #
    # step2_gan = DeepGAN(config=gan_config,
    #                     sess=sess,
    #                     data=step2_data,
    #                     g_config=g_config,
    #                     d_config=d_config)
    # step2_gan.train()
