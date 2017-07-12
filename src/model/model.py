class Model(object):
    def __init__(self, sess, data=None):
        self.sess = sess
        self.data = data
        self.name = 'Model'

    def train(self):
        pass

    def test(self):
        pass

    def create_model(self):
        pass

    def create_training_method(self):
        pass

    def save_model(self, model_path, epoch):
        self.model_saver.save(self.sess, model_path + 'model.ckpt', global_step=epoch)
        print('Model saved at %s model.ckpt' % model_path)

    def load_model(self, model_path):
        self.model_saver.restore(self.sess, model_path)
        print('Model at %s loaded' % model_path)
        pass

    def log_config(self, config_path):
        pass
