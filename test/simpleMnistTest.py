import os
import sys

import tensorflow as tf

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

sys.path.append(CURRENT_PATH)
sys.path.append(CURRENT_PATH + '/../')
from dataset import DATASET_PATH
from src.data.mnist.mnistConfig import MnistConfig
from src.data.mnist.mnistData import MnistData
from src.model.deepGAN.deepGAN import DeepGAN

if __name__ == '__main__':
    data = MnistData(data_path=DATASET_PATH + '/mnist', data_config=MnistConfig)
    sess = tf.InteractiveSession()
    gan = DeepGAN(sess=sess, data=data)
    gan.train()
    gan.log_config()
