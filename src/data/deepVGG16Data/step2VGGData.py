from src.data.vgg16.vgg16Data import VGG16Data
from dataset import DATASET_PATH
import numpy as np


class Step2VGGData(VGG16Data):
    def __init__(self, data_path, config, model_file):
        super(Step2VGGData, self).__init__(data_path, config, model_file, load_image=True)
        self.step2_z_data = self.load_step2_z_data()

    def return_z_batch_data(self, batch_size, index=None):
        # return super(Step2VGGData, self).return_z_batch_data(batch_size)
        # image = self.return_image_batch_data(batch_size, index=index)
        # res = self.eval_tensor_by_name(tensor_name=self.config.Z_SOURCE, image_batch=image)
        if index is None:
            index = 0

        res = self.step2_z_data[index * batch_size: (index + 1) * batch_size, ]

        return res

    def load_step2_z_data(self):
        data_path = DATASET_PATH + '/deepGANcat/step1_image/'
        step1_image_data = None
        for i in range(self.config.NPY_FILE_COUNT):
            res = np.load(data_path + 'step1_image' + 'batch_' + str(i) + '.npy')
            if i == 0:
                step1_image_data = res
            else:
                step1_image_data = np.concatenate((step1_image_data, res))
        return step1_image_data


if __name__ == '__main__':
    from dataset import DATASET_PATH
    from src.data.deepVGG16Data.step2VGGDataConfig import Step2VGGDataConfig
    import numpy as np

    d = Step2VGGData(data_path=DATASET_PATH + '/cat/',
                     config=Step2VGGDataConfig(),
                     model_file=DATASET_PATH + '/vgg16.tfmodel')
    print (d.return_z_batch_data(1, 0))
    # path = DATASET_PATH + '/deepGANcat/'

    # for i in range(100):
    #     image = d.return_image_batch_data(batch_size=10, index=i)
    #     np.save(file=path + 'step2_imagebatch_' + str(i), arr=image)
    #
    #     # data = np.load(file=path + 'step2_imagebatch_' + str(i) + '.npy').astype(np.uint8)
    #     # image = np.reshape(data[0,], newshape=[224, 224, 3]).astype(dtype=np.uint8)
    #     # im = Image.fromarray(image)
    #     # im.show()
    #
    #
    #     # image = np.reshape(image, newshape=[224, 224, 3]).astype(dtype=np.uint8)
    #     # im = Image.fromarray(image)
    #     # im.show()
