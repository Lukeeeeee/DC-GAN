from src.model.basicGAN.basicGAN import BasicGAN


class CatGAN(BasicGAN):
    def __init__(self, sess, data, config):
        super(CatGAN, self).__init__(sess=sess, data=data, config=config)
