import tensorflow as tf

from fmdcgan import MDCGAN

with tf.Session() as sess:
    model = MDCGAN(sess, in_height = 12, in_width = 12, in_channel = 512,
                 out_height = 24, out_width = 24, out_channel = 256,
                 dlearning_rate = 0.001, glearning_rate = 0.001,
                 sample_num = 1000,
                 data_real_name = 'relu3_1', data_z_name = 'relu4_1',
                 epoch_num = 400,
                 input_fname_pattern = '*.npy',
                 output_fname_pattern = '*.npy',
                 batch_size = 64)
    model.train()