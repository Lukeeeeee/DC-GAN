from src.data.vgg16.vgg16Data import VGG16Data


class Step2VGGData(VGG16Data):
    def __init__(self, data_path, config, model_file):
        super(Step2VGGData, self).__init__(data_path, config, model_file)

    def return_z_batch_data(self, batch_size, index=None):
        # return super(Step2VGGData, self).return_z_batch_data(batch_size)
        image = self.return_image_batch_data(batch_size, index=index)
        res = self.eval_tensor_by_name(tensor_name=self.config.Z_SOURCE, image_batch=image)
        return res


if __name__ == '__main__':
    from dataset import DATASET_PATH
    from src.data import DATA_PATH
    from src.data.deepVGG16Data.step2VGGDataConfig import Step2VGGDataConfig
    from PIL import Image
    import numpy as np

    d = Step2VGGData(data_path=DATASET_PATH + '/cat/',
                     config=Step2VGGDataConfig(),
                     model_file=DATA_PATH + '/vgg16/vgg16.tfmodel')
    image, _ = d.return_batch_data(batch_size=1, index=0)
    image = np.reshape(image, newshape=[224, 224, 3]).astype(dtype=np.uint8)
    im = Image.fromarray(image)
    im.show()
