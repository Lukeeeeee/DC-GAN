from src.data.vgg16.vgg16Data import VGG16Data


class Step1VGGData(VGG16Data):
    def __init__(self, data_path, config, model_file):
        super(Step1VGGData, self).__init__(data_path=data_path, config=config, model_file=model_file, load_image=False)
        self.step1_image_data = self.load_step1_image_data()

    def load_step1_image_data(self):
        data_path = DATASET_PATH + '/deepGANcat/step1_image/'
        step1_image_data = None
        for i in range(self.config.NPY_FILE_COUNT):
            res = np.load(data_path + 'step1_image' + 'batch_' + str(i) + '.npy')
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
    d = Step1VGGData(data_path=DATASET_PATH + '/cat/',
                     config=Step1VGGDataConfig(),
                     model_file=DATASET_PATH + '/vgg16.tfmodel')

    print(d.return_image_batch_data(1, 0))
    # print(d.return_batch_data(batch_size=1, index=0))
    path = DATASET_PATH + '/deepGANcat/'

    # for i in range(100):
    #     res = d.return_image_batch_data(batch_size=10, index=0)
    #     np.save(path + 'step1_image' + 'batch_' + str(i), res)
    #     # res = d.return_image_batch_data(batch_size=10, index=i)
    #     # data = np.load(path + 'step1_image' + 'batch_' + str(i) + '.npy')
    # pass
