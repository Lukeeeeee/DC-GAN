import tensorflow as tf

from fmdcgan import MDCGAN

with tf.Session() as sess:
    model = MDCGAN(sess, in_height = 96, in_width = 96, in_channel = 64,
                 out_height = 192, out_width = 192, out_channel = 3,
                 dlearning_rate = 0.001, glearning_rate = 0.001,
                 sample_num = 1000,
                 data_real_name = 'resize', data_z_name = 'relu1_1',
                 epoch_num = 400,
                 input_fname_pattern = '*.npy',
                 output_fname_pattern = '*.png',
                 batch_size = 64)
    model.train()