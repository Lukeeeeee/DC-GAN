import os
import sys

import tensorflow as tf

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

PARENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(CURRENT_PATH)
sys.path.append(PARENT_PATH)


from dataset import DATASET_PATH
from demo import DEMO_PATH
from src.data.mnist.mnistConfig import MnistConfig
from src.data.mnist.mnistData import MnistData
from src.model.basicGAN.basicGAN import BasicGAN
from src.model.basicGAN.ganConfig import GANConfig

if __name__ == '__main__':
    data = MnistData(data_path=DATASET_PATH + '/mnist', config=MnistConfig)
    sess = tf.InteractiveSession()
    gan_config = GANConfig()
    gan = BasicGAN(sess=sess, data=data, config=gan_config)

    # Train
    # gan.log_config()
    # gan.train()

    # Test

    gan.load_model(model_path=DEMO_PATH + '/7-16-23-39-45/model/', epoch=50)
    image_batch, z_batch = gan.data.return_batch_data(batch_size=gan.config.BATCH_SIZE,
                                                      index=1)
    res = gan.eval_tensor(tensor=gan.G.output, image_batch=image_batch, z_batch=z_batch)
    for i in range(10):
        gan.data.show_pic(data=res[i])
    for i in range(0):
        gan.data.show_pic(data=image_batch[i:i + 1, ])
