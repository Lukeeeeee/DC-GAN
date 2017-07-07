import math


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class Config(object):
    in_height = 1
    in_width = 1
    in_channel = 100
    out_height = 24
    out_width = 24
    out_channel = 256
    dlearning_rate = 1e-3
    glearning_rate = 0.00037
    sample_num = 5000
    data_real_name = 'relu3_1'
    data_z_name = None
    epoch_num = 400
    input_fname_pattern = '*.npy'
    output_fname_pattern = '*.npy'
    batch_size = 64

    s_h1, s_w1 = out_height, out_width
    s_h2, s_w2 = conv_out_size_same(s_h1, 2), conv_out_size_same(s_w1, 2)
    s_h3, s_w3 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
    s_h4, s_w4 = conv_out_size_same(s_h3, 2), conv_out_size_same(s_w3, 2)

    IN_HEIGHT = 1
    IN_WIDTH = 1
    IN_CHANNEL = 100

    OUT_HEIGHT = 24
    OUT_WIDTH = 24
    OUT_CHANNEL = 256

    D_LEARNING_RATE = 0.001
    G_LEARNING_RATE = 0, 003

    DATA_COUNT = 5000
    DATA_SOURCE = 'relu3_1'
    DATA_Z_NAME = None

    EPOCH = 100000
    BATCH_SIZE = 200
