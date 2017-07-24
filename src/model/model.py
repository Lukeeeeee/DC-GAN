class Model(object):
    def __init__(self, sess, config, data=None):
        self.sess = sess
        self.data = data
        self.name = 'Model'
        self.model_saver = None
        self.config = config

    def train(self):
        pass

    def test(self):
        pass

    def create_model(self):
        pass

    def create_training_method(self):
        pass

    def save_model(self, model_path, epoch):
        self.model_saver.save(self.sess,
                              save_path=model_path + 'model.ckpt',
                              global_step=epoch)
        print('Model saved at %s model.ckpt' % model_path)

    def load_model(self, model_path, epoch):
        self.model_saver.restore(sess=self.sess,
                                 save_path=model_path + 'model.ckpt-' + str(epoch))
        print('Model loaded at %s' % model_path)

    # def eval_tenor(self, tensor):
    #     pass

    def log_config(self):
        pass
