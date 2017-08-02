from src.data.vgg16.vgg16Data import VGG16Data
from dataset import DATASET_PATH
import numpy as np
from PIL import Image


class Step2VGGData(VGG16Data):
    def __init__(self, image_data_path, z_data_path, config, model_file, load_image=False):
        super(Step2VGGData, self).__init__(image_data_path, config, model_file, load_image=load_image)
        self.image_set = self.load_step2_image_data()
        print("finish load image data")
        self.z_data_path = z_data_path
        self.step2_z_data = self.load_step2_z_data()
        print("finish load z data")

    def load_step2_image_data(self):
        step2_image_data = None
        for i in range(self.config.NPY_FILE_COUNT):
            res = np.load(self.data_path + 'step1_image' + 'batch_' + str(i) + '.npy')
            if i == 0:
                step2_image_data = res
            else:
                step2_image_data = np.concatenate((step2_image_data, res))
        return step2_image_data

    def return_z_batch_data(self, batch_size, index=None):
        if index is None:
            index = 0

        res = self.step2_z_data[index * batch_size: (index + 1) * batch_size, ]
        return res

    def load_step2_z_data(self):
        step1_image_data = None
        for i in range(self.config.NPY_FILE_COUNT):
            res = np.load(self.z_data_path + 'step1_image' + 'batch_' + str(i) + '.npy')
            if i == 0:
                step1_image_data = res
            else:
                step1_image_data = np.concatenate((step1_image_data, res))
        return step1_image_data

if __name__ == '__main__':
    from dataset import DATASET_PATH
    import numpy as np
    import os
    from PIL import Image as im
    from test.vggDeepGANTest.tempCelebaconfig.step2.step2VGGCelebaDataConfig import Step2VGGCelebaDataConfig

    # path = os.path.join(DATASET_PATH, '../../celeba/original/')
    # path2 = os.path.join(DATASET_PATH, '../../celeba/reshape_224_224/')
    # for i in range(1, 202599 + 1):
    #     d = '{0:06}'.format(i)
    #     image = im.open(path + d + '.jpg')
    #     im_new = image.resize([224, 224], im.ANTIALIAS)
    #     im_new.save(path2 + d + '.jpg')
    #
    a = Step2VGGData(image_data_path=DATASET_PATH + '/celeba/224_224_3/',
                     z_data_path=DATASET_PATH + '/celeba/56_56_256/',
                     config=Step2VGGCelebaDataConfig(),
                     model_file=1,
                     load_image=False
                     )
    pass
