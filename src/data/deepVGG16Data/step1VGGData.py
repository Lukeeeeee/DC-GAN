from src.data.vgg16.vgg16Data import VGG16Data


class Step1VGGData(VGG16Data):
    def __init__(self, image_data_path, config, model_file):
        super(Step1VGGData, self).__init__(data_path=image_data_path, config=config, model_file=model_file,
                                           load_image=False)
        self.step1_image_data = self.load_step1_image_data()

    def load_step1_image_data(self):
        step1_image_data = None
        for i in range(self.config.NPY_FILE_COUNT):
            res = np.load(self.data_path + 'step1_image' + 'batch_' + str(i) + '.npy')
            if i == 0:
                step1_image_data = res
            else:
                step1_image_data = np.concatenate((step1_image_data, res))
        return step1_image_data

    def return_image_batch_data(self, batch_size, index):
        # image = super(Step1VGGData, self).return_image_batch_data(batch_size, index)
        # res = self.eval_tensor_by_name(tensor_name=self.config.IMAGE_SOURCE, image_batch=image)
        res = self.step1_image_data[index * batch_size: (index + 1) * batch_size, ]
        res = np.reshape(res,
                         newshape=[-1, self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_CHANNEL])
        return res


if __name__ == '__main__':
    from dataset import DATASET_PATH
    from src.data.deepVGG16Data.step1VGGDataConfig import Step1VGGDataConfig
    import numpy as np

    d = Step1VGGData(image_data_path=DATASET_PATH + '/cat/',
                     config=Step1VGGDataConfig(),
                     model_file=DATASET_PATH + '/vgg16.tfmodel')

    path = DATASET_PATH + '/deepGANcat/step1_image_56_56_256/'

    d.init_with_model(model_file=d.model_file)

    for i in range(100):
        # res = d.return_image_batch_data(batch_size=10, index=i)
        image = super(Step1VGGData, d).return_image_batch_data(batch_size=10, index=i)
        res = d.eval_tensor_by_name(tensor_name=d.config.IMAGE_SOURCE, image_batch=image)
        res = np.reshape(a=res, newshape=[-1, 56, 56, 256])
        np.save(path + 'step1_imagebatch_' + str(i), res)
        # res = d.return_image_batch_data(batch_size=10, index=i)
        # data = np.load(path + 'step1_image_28_28_256' + 'batch_' + str(i) + '.npy')
    pass
