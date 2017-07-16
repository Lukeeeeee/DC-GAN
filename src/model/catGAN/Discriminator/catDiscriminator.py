from src.model.basicGAN.Discriminator.discriminator import Discriminator


class CatDiscriminator(Discriminator):
    def __init__(self, sess, data, config, generator):
        super(CatDiscriminator, self).__init__(sess=sess, data=data, config=config, generator=generator)
