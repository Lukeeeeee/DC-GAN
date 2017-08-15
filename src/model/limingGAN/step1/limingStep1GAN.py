from src.model.model import Model
from src.model.limingGAN.step1.src import *
import tensorflow as tf


class LimingStep1GAN(Model):
    def __init__(self, sess, config, data):
        super(LimingStep1GAN, self).__init__(sess=sess, config=config, data=data)
        self.name = 'LIMING_STEP1_GAN'

        self.noise_input = tf.placeholder(tf.float32, shape=[None, Noise_h, Noise_w, Noise_ch], name='noise')
        self.noise_sample_input = tf.placeholder(tf.float32, shape=[None, Noise_h, Noise_w, Noise_ch], name='noise')
        self.image_input = tf.placeholder(tf.float32, shape=[None, Image_h, Image_w, Image_ch], name='image')
        self.G = generate(z=noise_input, h=Image_h, w=Image_w, ch=Image_ch, is_training=True, reuse=None,
                          batch_size=Batch_size)
        # param of G
        self.G_vars = tf.trainable_variables()
        self.D = decrim(image_input, True, None, batch_size=Batch_size)

        self.D_vars = []
        for item in tf.trainable_variables():
            if item not in self.G_vars:
                self.D_vars.append(item)

        self.d_real = self.D
        self.d_fake = decrim(G, True, True)

        self.D_loss = loss_train_D = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real, labels=tf.ones_like(self.d_real))) \
                                     + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.zeros_like(self.d_fake)))
        tf.summary.scalar('d_loss', loss_train_D)

        self.G_loss = loss_train_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.ones_like(self.d_fake)))
        tf.summary.scalar('g_loss', loss_train_G)
        self.g_optimizer = self.optimizer(loss_train_G, G_learnrate, G_vars, name='opt_train_G')
        self.d_optimizer = self.optimizer(loss_train_D, D_learnrate, D_vars, name='opt_train_D')
        self.saver = tf.train.Saver()

    @staticmethod
    def optimizer(loss, learning_rate, vlist=None, name=None):
        with tf.variable_scope(name):
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name=name + '/Adam')
            return opt.minimize(loss, var_list=vlist, name=name + '/opt')

    def train(self):
        count = 0
        for i in range(self.config.TRAINING_EPOCH):
            D_aver_loss = 0.0
            G_aver_loss = 0.0
            for j in range(self.config.BATCH_COUNT):
                count = count + 1
                image_batch, z_batch = self.data.return_batch_data(batch_size=self.config.BATCH_SIZE,
                                                                   index=j)
                D_loss = self.update_discriminator(image_batch=image_batch,
                                                   z_batch=z_batch)
                G_loss_1 = self.update_generator(z_batch=z_batch)
                # G_loss_2 = self.update_generator(z_batch=z_batch)

                G_loss_2 = G_loss_1
                G_loss = (G_loss_1 + G_loss_2) / 2.0

                D_aver_loss = (D_aver_loss * float(j) + D_loss) / float(j + 1)

                G_aver_loss = (G_aver_loss * float(j) + G_loss) / float(j + 1)

                print("epoch: %d batch: %d  gloss:%.4f dloss:%.4f" %
                      (e + 1, idx, g_loss, d_loss))

            image_batch, z_batch = self.data.return_batch_data(batch_size=self.config.BATCH_SIZE,
                                                               index=0)
            self.run_summary(image_batch=image_batch, z_batch=z_batch, count=i)
            print("Print D real res")
            print(self.eval_tensor(tensor=self.D.real_D, image_batch=image_batch, z_batch=z_batch))
            print("Print D fake res")
            print(self.eval_tensor(tensor=self.D.fake_D, image_batch=image_batch, z_batch=z_batch))

            if (i + 1) % self.config.SAVE_MODEL_EVERY_EPOCH == 0:
                self.save_model(model_path=self.model_dir, epoch=i + 1)
