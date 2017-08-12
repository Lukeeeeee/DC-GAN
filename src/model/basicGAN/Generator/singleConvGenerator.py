import tensorflow as tf

from src.common.ops import ops
from src.model.model import Model


class SingleConvGenerator(Model):
    def __init__(self, sess, data, config):
        super(SingleConvGenerator, self).__init__(sess=sess, data=data, config=config)
        self.name = 'Generator'
        self.variable_dict = {
            'W_1': tf.Variable(tf.truncated_normal(shape=[self.config.FILTER_SIZE,
                                                          self.config.FILTER_SIZE,
                                                          self.config.LAYER_1_OUT_CHANNEL,
                                                          self.config.IN_CHANNEL],
                                                   stddev=self.config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                               name='W_1'),

            'B_1': tf.Variable(tf.constant(value=0.0,
                                           shape=[self.config.LAYER_1_OUT_CHANNEL]),
                               name='B_1'),

            'BETA_1': tf.Variable(tf.constant(value=0.0,
                                              shape=[self.config.LAYER_1_OUT_CHANNEL]),
                                  name='BETA_1'),
            'GAMMA_1': tf.Variable(tf.random_normal(shape=[self.config.LAYER_1_OUT_CHANNEL]),
                                   name='GAMMA_1'),
            'W_2': tf.Variable(tf.truncated_normal(shape=[self.config.FILTER_SIZE,
                                                          self.config.FILTER_SIZE,
                                                          self.config.OUT_CHANNEL,
                                                          self.config.LAYER_1_OUT_CHANNEL],
                                                   stddev=self.config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                               name='W_2'),

            'B_2': tf.Variable(tf.constant(value=0.0,
                                           shape=[self.config.OUT_CHANNEL]),
                               name='B_2'),

            'BETA_2': tf.Variable(tf.constant(value=0.0,
                                              shape=[self.config.OUT_CHANNEL]),
                                  name='BETA_2'),
            'GAMMA_2': tf.Variable(tf.random_normal(shape=[self.config.OUT_CHANNEL]),
                                   name='GAMMA_2')
        }
        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.config.IN_WIDTH, self.config.IN_HEIGHT,
                                           self.config.IN_CHANNEL],
                                    name='G_INPUT')
        self.input_list = tf.unstack(value=self.input, axis=3, name='UNSTACK_INPUT')

        self._var_list = []
        self.var_summary_list = []

        self._loss = None
        self.optimizer = None
        self.optimize_loss = None
        self.gradients = None

        self.is_training = tf.placeholder(tf.bool)
        self.output = self.create_model()
        self.output_after_transfer = self.transfer_to_pic(tensor=self.output)
        with tf.name_scope('GeneratorOutput'):
            pic_summary = tf.summary.image(name='Genrator_Step3_Output',
                                           tensor=self.output_after_transfer,
                                           max_outputs=50)

        self.var_summary_list.append(pic_summary)
        for key, value in self.variable_dict.iteritems():
            self.var_list.append(value)
            with tf.name_scope('G_weight_summary'):
                summary = tf.summary.tensor_summary(value.op.name, value)
                histogram = tf.summary.histogram(value.op.name, value)
                self.var_summary_list.append(summary)
                self.var_summary_list.append(histogram)

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, new_loss):
        self._loss = new_loss
        self.optimizer, self.gradients, self.optimize_loss = self.create_training_method()
        # ops.variable_summaries(self.gradients)
        ops.variable_summaries(self.loss)
        with tf.name_scope('G_gradients'):
            for node in self.gradients:
                try:
                    summary = tf.summary.tensor_summary('test', node[0])
                    histogram = tf.summary.histogram('test', node[0])
                    self.var_summary_list.append(summary)
                    self.var_summary_list.append(histogram)
                except BaseException:
                    print("wrong")

    @property
    def var_list(self):
        return self._var_list

    @var_list.setter
    def var_list(self, new_list):
        self._var_list = new_list
        # self.optimizer, self.gradients, self.optimize_loss = self.create_training_method()

    def create_model(self):
        tran_conv_1 = tf.nn.conv2d_transpose(value=self.input,
                                             filter=self.variable_dict['W_1'],
                                             output_shape=[self.config.BATCH_SIZE,
                                                           self.config.LAYER_1_OUT_WIDTH,
                                                           self.config.LAYER_1_OUT_HEIGHT,
                                                           self.config.LAYER_1_OUT_CHANNEL],
                                             strides=[1, self.config.CONV_1_STRIDE, self.config.CONV_1_STRIDE, 1],
                                             padding='SAME')
        tran_conv_1 = tf.nn.bias_add(tran_conv_1, self.variable_dict['B_1'])

        tran_conv_1 = ops.batch_norm(x=tran_conv_1,
                                     beta=self.variable_dict['BETA_1'],
                                     gamma=self.variable_dict['GAMMA_1'],
                                     phase_train=self.is_training,
                                     scope='BATCH_NORM_1')
        tran_conv_1 = tf.nn.relu(tran_conv_1, name='RELU')

        tran_conv_2 = tf.nn.conv2d_transpose(value=tran_conv_1,
                                             filter=self.variable_dict['W_2'],
                                             output_shape=[self.config.BATCH_SIZE,
                                                           self.config.OUT_WIDTH,
                                                           self.config.OUT_HEIGHT,
                                                           self.config.OUT_CHANNEL],
                                             strides=[1, self.config.CONV_2_STRIDE, self.config.CONV_2_STRIDE, 1],
                                             padding='SAME')
        tran_conv_2 = tf.nn.bias_add(tran_conv_2, self.variable_dict['B_2'])

        tran_conv_2 = ops.batch_norm(x=tran_conv_2,
                                     beta=self.variable_dict['BETA_2'],
                                     gamma=self.variable_dict['GAMMA_2'],
                                     phase_train=self.is_training,
                                     scope='BATCH_NORM_2')

        tran_conv_2 = tf.nn.tanh(tran_conv_2, name='TANH')

        return tran_conv_2

    def create_training_method(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.G_LEARNING_RATE)
        gradients = optimizer.compute_gradients(loss=self.loss, var_list=self.var_list)
        optimize_loss = optimizer.minimize(loss=self.loss, var_list=self.var_list)

        return optimizer, gradients, optimize_loss
