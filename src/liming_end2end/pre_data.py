'''
api of layers
'''
import os
import numpy as np
from glob import glob
from PIL import Image
from dataset import DATASET_PATH
import scipy.io as sio


def get_datalist():
    # mage_list = glob(os.path.join(data_dir, data_pattern))
    data_path = DATASET_PATH + '/mnist/mnist_7_7_16/'
    z_data = None
    for i in range(60000):
        res = np.random.uniform(0, 1, size=[1, 1, 100])
        if i == 0:
            z_data = res
        else:
            z_data = np.concatenate((z_data, res))

    data_path = DATASET_PATH + '/mnist/'
    image_data = None
    for i in range(10):
        mat_file_path = data_path + '/digit' + str(i) + '.mat'
        data = sio.loadmat(mat_file_path)
        data = np.array(data['D'])
        if i == 0:
            image_data = data
        else:
            image_data = np.concatenate((image_data, data))
    image_data = np.reshape(np.ravel(image_data,
                                     order='C'),
                            newshape=[-1, 28, 28, 1],
                            ).astype(np.float32)
    np.random.shuffle(image_data)
    np.random.shuffle(z_data)
    return image_data, z_data


def get_image(image_list, batch_size, img_h, img_w):
    image_batch = []
    for img in image_list:
        data = Image.open(img)
        data = data.resize((img_h, img_w))
        data = np.array(data)
        data = data.astype('float32') / 127.5 - 1
        image_batch.append(data)
    return (image_batch)


def restruct_image(x, batch_size):
    image_batch = []
    for k in range(batch_size):
        data = x[k, :, :, :]
        data = (data + 1) * 127.5
        # data = np.clip(data,0,255).astype(np.uint8)
        image_batch.append(data)
    return (image_batch)


if __name__ == '__main__':
    get_datalist()
