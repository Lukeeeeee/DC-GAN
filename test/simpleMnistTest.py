import tensorflow as tf

from dataset import DATASET_PATH
from src.data.mnist.mnistData import MnistData
from src.model.deepGAN.deepGAN import DeepGAN

if __name__ == '__main__':
    data = MnistData(data_path=DATASET_PATH + '/mnist')
    sess = tf.InteractiveSession()
    gan = DeepGAN(sess=sess, data=data)
    gan.train()
