from src.model.basicGAN.Generator.generator import Generator


class CatGenerator(Generator):
    def __init__(self, sess, data, config):
        super(CatGenerator, self).__init__(sess=sess, data=data, config=config)
