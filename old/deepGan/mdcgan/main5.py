import tensorflow as tf

from fmdcgan import MDCGAN

with tf.Session() as sess:
    model = MDCGAN(sess, in_height = 48, in_width = 48, in_channel = 128,
                 out_height = 96, out_width = 96, out_channel = 64,
                 dlearning_rate = 0.001, glearning_rate = 0.001,
                 sample_num = 1000,
                 data_real_name = 'relu1_1', data_z_name = 'relu2_1',
                 epoch_num = 400,
                 input_fname_pattern = '*.npy',
                 output_fname_pattern = '*.npy',
                 batch_size = 64)
    model.train()