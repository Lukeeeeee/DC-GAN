'''
api of layers
'''
import numpy as np
from PIL import Image
from dataset import DATASET_PATH
from src.liming_step1.train import Noise_w, Noise_h, Noise_ch, Sample_num


def load_celeba_image(path):
    image_data = None
    for i in range(10):
        res = np.load(path + str(i) + '.npy')
        if i == 0:
            image_data = res
        else:
            image_data = np.concatenate((image_data, res))
    np.random.shuffle(image_data)
    return image_data


def get_datalist():
    image_data = load_celeba_image(DATASET_PATH + '/celeba/224_224_3/')
    image_14_data = load_celeba_image(DATASET_PATH + '/celeba/14_14_512/')
    image_28_data = load_celeba_image(DATASET_PATH + '/celeba/28_28_256/')
    image_56_data = load_celeba_image(DATASET_PATH + '/celeba/56_56_128/')
    image_112_data = load_celeba_image(DATASET_PATH + '/celeba/112_112_64/')

    z_data = None
    for i in range(Sample_num):
        data = np.random.normal(0, 1, [1, Noise_h, Noise_w, Noise_ch])
        if i == 0:
            z_data = data
        else:
            z_data = np.concatenate((z_data, data))

    return image_data, image_14_data, image_28_data, image_56_data, image_112_data, z_data


def get_image(image_list, batch_size, img_h, img_w):
    image_batch = []
    for img in image_list:
        data = Image.open(img)
        data = data.resize((img_h, img_w))
        data = np.array(data)
        data = data.astype('float32') / 127.5 - 1
        image_batch.append(data)
    return image_batch


def restruct_image(x, batch_size):
    image_batch = []
    for k in range(batch_size):
        data = x[k, :, :, :]
        data = (data + 1) * 127.5
        # data = np.clip(data,0,255).astype(np.uint8)
        image_batch.append(data)
    return image_batch


if __name__ == '__main__':
    get_datalist()
