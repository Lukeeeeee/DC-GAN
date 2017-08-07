import tensorflow as tf

from dataset import DATASET_PATH
from demo import DEMO_PATH
from src.data.deepGAN import *
from src.data.mnistCNN import *
from src.model.deepGAN.deepGAN import DeepGAN
from src.model.mnistCNN import *
from test.mnistDeepGANTest.tempConfig.step1 import *
from test.mnistDeepGANTest.tempConfig.step2 import *
from test.mnistDeepGANTest.tempConfig.step3 import *


def return_step1_gan(mnist_cnn):
    data_sess = tf.InteractiveSession()
    config = Step1DataConfig()
    step1_data = Step1Data(data_path=DATASET_PATH + '/mnist',
                           config=config,
                           sess=data_sess,
                           mnist_cnn=mnist_cnn)

    step1_sess = tf.InteractiveSession()
    gan_config = Step1GANConfig()
    d_config = Step1DiscriminatorConfig()
    g_config = Step1GeneratorConfig()
    step1_gan = DeepGAN(config=gan_config,
                        sess=step1_sess,
                        data=step1_data,
                        g_config=g_config,
                        d_config=d_config)
    return step1_gan


def return_step2_gan(mnist_cnn):
    data_sess = tf.InteractiveSession()
    config = Step2DataConfig()
    step2_data = Step2Data(data_path=DATASET_PATH + '/mnist',
                           config=config,
                           sess=data_sess,
                           mnist_cnn=mnist_cnn)

    step2_sess = tf.InteractiveSession()
    gan_config = Step2GANConfig()
    d_config = Step2DiscriminatorConfig()
    g_config = Step2GeneratorConfig()
    step2_gan = DeepGAN(config=gan_config,
                        sess=step2_sess,
                        data=step2_data,
                        g_config=g_config,
                        d_config=d_config)
    return step2_gan


def retutn_step3_gan(mnist_cnn):
    data_sess = tf.InteractiveSession()
    config = Step3DataConfig()
    step2_data = Step3Data(data_path=DATASET_PATH + '/mnist',
                           config=config,
                           sess=data_sess,
                           mnist_cnn=mnist_cnn)

    step2_sess = tf.InteractiveSession()
    gan_config = Step3GANConfig()
    d_config = Step3DiscriminatorConfig()
    g_config = Step3GeneratorConfig()
    step2_gan = DeepGAN(config=gan_config,
                        sess=step2_sess,
                        data=step2_data,
                        g_config=g_config,
                        d_config=d_config,
                        step2_flag=False,
                        single_flag=True)
    return step2_gan


def test(step1_gan, step2_gan):
    step1_gan.load_model(model_path=DEMO_PATH + '/7-25-9-22-22/Step1_GAN/model/', epoch=10)
    step2_gan.load_model(model_path=DEMO_PATH + '/7-25-9-22-57/Step2_GAN/model/', epoch=10)
    image_batch, z_batch = step1_gan.data.return_batch_data(200, 0)
    mid_data = step1_gan.eval_tensor(tensor=step1_gan.G.output,
                                     image_batch=image_batch,
                                     z_batch=z_batch)
    image_batch, _ = step2_gan.data.return_batch_data(200, 0)
    res = step2_gan.eval_tensor(tensor=step2_gan.G.output,
                                image_batch=image_batch,
                                z_batch=mid_data)
    for i in range(10):
        step1_gan.data.show_pic(data=res[i * 10])


if __name__ == '__main__':
    mnist_cnn_sess = tf.InteractiveSession()

    config = MnistCNNDataConfig()
    data = MnistCNNData(data_path=DATASET_PATH + '/mnist/', config=config)

    config = MnistCNNConfig()
    mnist_cnn = MnistCNN(config=config, sess=mnist_cnn_sess, data=data)
    mnist_cnn.load_model(model_path=DEMO_PATH + '/mnist/7-18-18-34-45/model/', epoch=3)
    step3 = retutn_step3_gan(mnist_cnn)
    step3.train()
