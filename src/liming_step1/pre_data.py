'''
api of layers
'''
import os
import numpy as np
from glob import glob
from PIL import Image
from dataset import DATASET_PATH
import scipy.io as sio
import numpy as np

def get_datalist():
    # mage_list = glob(os.path.join(data_dir, data_pattern))
    data_path = DATASET_PATH + '/mnist/mnist_7_7_16/'
    image_data = None
    for i in range(10):
        res = np.load(data_path + str(i) + '.npy')
        if i == 0:
            image_data = res
        else:
            image_data = np.concatenate((image_data, res))

    data_path = DATASET_PATH + '/mnist/'
    z_data = None
    for i in range(200):
        data = np.random.normal(0, 1, [1, 1, 1, 100])
        if i == 0:
            z_data = data
        else:
            z_data = np.concatenate((z_data, data))

    np.random.shuffle(image_data)
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
