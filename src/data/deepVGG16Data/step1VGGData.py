from src.data.vgg16.vgg16Data import VGG16Data


class Step1VGGData(VGG16Data):
    def __init__(self, data_path, config, model_file):
        super(Step1VGGData, self).__init__(data_path=data_path, config=config, model_file=model_file)

    def return_image_batch_data(self, batch_size, index):
        image = super(Step1VGGData, self).return_image_batch_data(batch_size, index)
        res = self.eval_tensor_by_name(tensor_name=self.config.IMAGE_SOURCE, image_batch=image)
        return res


if __name__ == '__main__':
    from dataset import DATASET_PATH
    from src.data import DATA_PATH
    from src.data.deepVGG16Data.step1VGGDataConfig import Step1VGGDataConfig

    d = Step1VGGData(data_path=DATASET_PATH + '/cat/',
                     config=Step1VGGDataConfig(),
                     model_file=DATA_PATH + '/vgg16/vgg16.tfmodel')
    print(d.return_batch_data(batch_size=1, index=0))
