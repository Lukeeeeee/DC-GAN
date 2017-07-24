import tensorflow as tf

from src.common.ops import ops
from src.model.model import Model


class Generator(Model):
    def __init__(self, sess, data, config):
        super(Generator, self).__init__(sess=sess, data=data, config=config)
        self.name = 'Generator'
        with tf.variable_scope(self.name):
            self.variable_dict = {
                'W_1': tf.Variable(tf.truncated_normal(shape=([self.config.IN_CHANNEL *
                                                               self.config.IN_WIDTH *
                                                               self.config.IN_HEIGHT,
                                                               self.config.TRAN_CONV_LAYER_1_IN_CHANNEL *
                                                               self.config.TRAN_CONV_LAYER_1_HEIGHT *
                                                               self.config.TRAN_CONV_LAYER_1_WIDTH]),
                                                       stddev=self.config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                                   name='W_1'),
                'B_1': tf.Variable(tf.constant(value=0.0,
                                               shape=[self.config.TRAN_CONV_LAYER_1_IN_CHANNEL *
                                                      self.config.TRAN_CONV_LAYER_1_WIDTH *
                                                      self.config.TRAN_CONV_LAYER_1_HEIGHT]),
                                   name='B_1'),
                'BETA_1': tf.Variable(tf.truncated_normal(shape=[self.config.TRAN_CONV_LAYER_1_IN_CHANNEL]),
                                      name='BETA_1'),

                'GAMMA_1': tf.Variable(tf.random_normal(shape=[self.config.TRAN_CONV_LAYER_1_IN_CHANNEL],
                                                        mean=self.config.BATCH_NORM_MEAN,
                                                        stddev=self.config.BATCH_STANDARD_DEVIATION),
                                       name='GAMMA_1'),

                'W_2': tf.Variable(tf.truncated_normal(shape=[self.config.FILTER_SIZE,
                                                              self.config.FILTER_SIZE,
                                                              self.config.TRAN_CONV_LAYER_2_IN_CHANNEL,
                                                              self.config.TRAN_CONV_LAYER_1_IN_CHANNEL],
                                                       stddev=self.config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                                   name='W_2'),

                'B_2': tf.Variable(tf.constant(value=0.0,
                                               shape=[self.config.TRAN_CONV_LAYER_2_IN_CHANNEL]),
                                   name='B_2'),

                'BETA_2': tf.Variable(tf.constant(value=0.0,
                                                  shape=[self.config.TRAN_CONV_LAYER_2_IN_CHANNEL]),
                                      name='BETA_2'),
                'GAMMA_2': tf.Variable(tf.random_normal(shape=[self.config.TRAN_CONV_LAYER_2_IN_CHANNEL]),
                                       name='GAMMA_2'),

                'W_3': tf.Variable(tf.truncated_normal(shape=[self.config.FILTER_SIZE,
                                                              self.config.FILTER_SIZE,
                                                              self.config.TRAN_CONV_LAYER_3_IN_CHANNEL,
                                                              self.config.TRAN_CONV_LAYER_2_IN_CHANNEL],
                                                       stddev=self.config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                                   name='W_3'),

                'B_3': tf.Variable(tf.constant(value=0.0,
                                               shape=[self.config.TRAN_CONV_LAYER_3_IN_CHANNEL]),
                                   name='B_3'),

                'BETA_3': tf.Variable(tf.constant(value=0.0,
                                                  shape=[self.config.TRAN_CONV_LAYER_3_IN_CHANNEL]),
                                      name='BETA_3'),

                'GAMMA_3': tf.Variable(tf.random_normal(shape=[self.config.TRAN_CONV_LAYER_3_IN_CHANNEL]),
                                       name='GAMMA_3'),

                'W_4': tf.Variable(tf.truncated_normal(shape=[self.config.FILTER_SIZE,
                                                              self.config.FILTER_SIZE,
                                                              self.config.OUT_CHANNEL,
                                                              self.config.TRAN_CONV_LAYER_3_IN_CHANNEL],
                                                       stddev=self.config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                                   name='W_4'),
                'B_4': tf.Variable(tf.constant(value=0.0,
                                               shape=[self.config.OUT_CHANNEL]),
                                   name='B_4'),

                'BETA_4': tf.Variable(tf.constant(value=0.0,
                                                  shape=[self.config.OUT_CHANNEL]),
                                      name='BETA_4'),

                'GAMMA_4': tf.Variable(tf.random_normal(shape=[self.config.OUT_CHANNEL],
                                                        mean=self.config.BATCH_NORM_MEAN,
                                                        stddev=self.config.BATCH_STANDARD_DEVIATION),
                                       name='GAMMA_4')

            }
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=[None, self.config.IN_WIDTH, self.config.IN_HEIGHT,
                                               self.config.IN_CHANNEL],
                                        name='G_INPUT')
            self._loss = None

            self.is_training = tf.placeholder(tf.bool)
        self.output = self.create_model()
        self.var_list = []
        for key, value in self.variable_dict.iteritems():
            self.var_list.append(value)
            # self.optimizer, self.gradients, self.optimize_loss = self.create_training_method()

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, new_loss):
        self._loss = new_loss
        self.optimizer, self.gradients, self.optimize_loss = self.create_training_method()
        # ops.variable_summaries(self.gradients)
        self.loss_scalar_summary, self.loss_histogram_summary = ops.variable_summaries(self.loss)

    def create_model(self):
        with tf.variable_scope('Generator', reuse=False):
            input = tf.reshape(tensor=self.input,
                               shape=[-1, self.config.IN_HEIGHT * self.config.IN_WIDTH * self.config.IN_CHANNEL])
            tran_fc = tf.add(tf.matmul(input, self.variable_dict['W_1']), self.variable_dict['B_1'])
            tran_fc = tf.reshape(tensor=tran_fc,
                                 shape=[-1, self.config.TRAN_CONV_LAYER_1_WIDTH, self.config.TRAN_CONV_LAYER_1_HEIGHT,
                                        self.config.TRAN_CONV_LAYER_1_IN_CHANNEL])
            tran_fc = ops.batch_norm(x=tran_fc,
                                     beta=self.variable_dict['BETA_1'],
                                     gamma=self.variable_dict['GAMMA_1'],
                                     phase_train=self.is_training,
                                     scope='BATCH_NORM_1')
            tran_fc = tf.nn.relu(tran_fc, name='RELU_1')

            tran_conv_1 = tf.nn.conv2d_transpose(value=tran_fc,
                                                 filter=self.variable_dict['W_2'],
                                                 output_shape=[self.config.BATCH_SIZE,
                                                               self.config.TRAN_CONV_LAYER_2_WIDTH,
                                                               self.config.TRAN_CONV_LAYER_2_HEIGHT,
                                                               self.config.TRAN_CONV_LAYER_2_IN_CHANNEL],
                                                 strides=[1, self.config.CONV_STRIDE, self.config.CONV_STRIDE, 1],
                                                 padding='SAME')
            tran_conv_1 = tf.nn.bias_add(tran_conv_1, self.variable_dict['B_2'])

            tran_conv_1 = ops.batch_norm(x=tran_conv_1,
                                         beta=self.variable_dict['BETA_2'],
                                         gamma=self.variable_dict['GAMMA_2'],
                                         phase_train=self.is_training,
                                         scope='BATCH_NORM_2')
            tran_conv_1 = tf.nn.relu(tran_conv_1, name='RELU_2')

            tran_conv_2 = tf.nn.conv2d_transpose(value=tran_conv_1,
                                                 filter=self.variable_dict['W_3'],
                                                 output_shape=[self.config.BATCH_SIZE,
                                                               self.config.TRAN_CONV_LAYER_3_WIDTH,
                                                               self.config.TRAN_CONV_LAYER_3_HEIGHT,
                                                               self.config.TRAN_CONV_LAYER_3_IN_CHANNEL],
                                                 strides=[1, self.config.CONV_STRIDE, self.config.CONV_STRIDE, 1],
                                                 padding='SAME')
            tran_conv_2 = tf.nn.bias_add(tran_conv_2, self.variable_dict['B_3'])

            tran_conv_2 = ops.batch_norm(x=tran_conv_2,
                                         beta=self.variable_dict['BETA_3'],
                                         gamma=self.variable_dict['GAMMA_3'],
                                         phase_train=self.is_training,
                                         scope='BATCH_NORM_3')

            tran_conv_2 = tf.nn.relu(tran_conv_2, name='RELU_3')

            tran_conv_3 = tf.nn.conv2d_transpose(value=tran_conv_2,
                                                 filter=self.variable_dict['W_4'],
                                                 output_shape=[self.config.BATCH_SIZE,
                                                               self.config.OUT_HEIGHT,
                                                               self.config.OUT_WIDTH,
                                                               self.config.OUT_CHANNEL],
                                                 strides=[1, self.config.CONV_STRIDE, self.config.CONV_STRIDE, 1],
                                                 padding='SAME')
            tran_conv_3 = tf.nn.bias_add(tran_conv_3, self.variable_dict['B_4'])

            tran_conv_3 = ops.batch_norm(x=tran_conv_3,
                                         beta=self.variable_dict['BETA_4'],
                                         gamma=self.variable_dict['GAMMA_4'],
                                         phase_train=self.is_training,
                                         scope='BATCH_NORM_4')
            tran_conv_3 = tf.nn.tanh(tran_conv_3, name='RELU_4')

            return tran_conv_3

    def create_training_method(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.G_LEARNING_RATE)
        gradients = optimizer.compute_gradients(loss=self.loss, var_list=self.var_list)
        optimize_loss = optimizer.minimize(loss=self.loss, var_list=self.var_list)

        return optimizer, gradients, optimize_loss

        # def update(self, z_batch):
        #     loss, _ = self.sess.run(fetches=[self.loss, self.optimize_loss],
        #                             feed_dict={self.input: z_batch,
        #                                        self.is_training: True,
        #                                        self.discriminator.is_training: True})
        #     return loss
