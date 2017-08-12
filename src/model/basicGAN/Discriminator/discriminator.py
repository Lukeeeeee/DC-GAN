from __future__ import print_function
import tensorflow as tf

from src.common.ops import ops
from src.model.model import Model


class Discriminator(Model):
    def __init__(self, sess, data, generator, config):
        super(Discriminator, self).__init__(sess=sess, data=data, config=config)
        self.name = 'Discriminator'
        self.var_summary_list = []
        self.variable_dict = {
            "W_1": tf.Variable(tf.truncated_normal([self.config.FILTER_SIZE, self.config.FILTER_SIZE,
                                                    self.config.IN_CHANNEL,
                                                    self.config.CONV_LAYER_1_OUT_CHANNEL],
                                                   stddev=self.config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                               name='W_1'),

            "B_1": tf.Variable(tf.constant(0.0, shape=[self.config.CONV_LAYER_1_OUT_CHANNEL]),
                               name='B_1'),

            'BETA_1': tf.Variable(tf.constant(0.0, shape=[self.config.CONV_LAYER_1_OUT_CHANNEL]),
                                  name='BETA_1'),

            'GAMMA_1': tf.Variable(tf.random_normal(shape=[self.config.CONV_LAYER_1_OUT_CHANNEL],
                                                    mean=self.config.BATCH_NORM_MEAN,
                                                    stddev=self.config.BATCH_STANDARD_DEVIATION),
                                   name='GAMMA_1'),

            "W_2": tf.Variable(tf.truncated_normal([self.config.FILTER_SIZE, self.config.FILTER_SIZE,
                                                    self.config.CONV_LAYER_1_OUT_CHANNEL,
                                                    self.config.CONV_LAYER_2_OUT_CHANNEL],
                                                   stddev=self.config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                               name='W_2'),

            "B_2": tf.Variable(tf.constant(0.0, shape=[self.config.CONV_LAYER_2_OUT_CHANNEL]),
                               name='B_2'),

            'BETA_2': tf.Variable(tf.constant(0.0, shape=[self.config.CONV_LAYER_2_OUT_CHANNEL]),
                                  name='BETA_2'),

            'GAMMA_2': tf.Variable(tf.random_normal(shape=[self.config.CONV_LAYER_2_OUT_CHANNEL],
                                                    mean=self.config.BATCH_NORM_MEAN,
                                                    stddev=self.config.BATCH_STANDARD_DEVIATION),
                                   name='GAMMA_2'),

            "W_3": tf.Variable(tf.truncated_normal([self.config.FILTER_SIZE, self.config.FILTER_SIZE,
                                                    self.config.CONV_LAYER_2_OUT_CHANNEL,
                                                    self.config.CONV_LAYER_3_OUT_CHANNEL],
                                                   stddev=self.config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                               name='W_3'),

            "B_3": tf.Variable(tf.constant(0.0, shape=[self.config.CONV_LAYER_3_OUT_CHANNEL]),
                               name='B_3'),

            'BETA_3': tf.Variable(tf.constant(0.0, shape=[self.config.CONV_LAYER_3_OUT_CHANNEL]),
                                  name='BETA_3'),

            'GAMMA_3': tf.Variable(tf.random_normal(shape=[self.config.CONV_LAYER_3_OUT_CHANNEL],
                                                    mean=self.config.BATCH_NORM_MEAN,

                                                    stddev=self.config.BATCH_STANDARD_DEVIATION),
                                   name='GAMMA_3'),

            "W_4": tf.Variable(tf.truncated_normal([(self.config.CONV_OUT_HEIGHT *
                                                     self.config.CONV_OUT_WIDTH *
                                                     self.config.CONV_LAYER_3_OUT_CHANNEL),
                                                    self.config.OUTPUT_SIZE],
                                                   stddev=self.config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                               name='W_4'),

            "B_4": tf.Variable(tf.constant(0.0, shape=[self.config.OUTPUT_SIZE]), name='b_4')
        }

        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.config.IN_WIDTH,
                                           self.config.IN_HEIGHT, self.config.IN_CHANNEL],
                                    name='D_INPUT')
        self.is_training = tf.placeholder(tf.bool)
        self.input_transfer_to_pic = self.transfer_to_pic(tensor=self.input)
        with tf.name_scope('OriginalPic'):
            pic_summary = tf.summary.image(name='OriginalPic',
                                           tensor=self.input_transfer_to_pic,
                                           max_outputs=50)
        self.var_summary_list.append(pic_summary)

        for key, value in self.variable_dict.iteritems():
            with tf.name_scope('D_weight_summary'):
                summary = tf.summary.tensor_summary(value.op.name, value)
                histogram = tf.summary.histogram(value.op.name, value)
                self.var_summary_list.append(summary)
                self.var_summary_list.append(histogram)

        self._var_list = []

        self.generator = generator

        self.real_D, self.real_D_logits = self.create_model(input=self.input)

        self.real_D_predication = tf.round(self.real_D)

        self.fake_D, self.fake_D_logits = self.create_model(input=self.generator.output)

        self.fake_D_predication = tf.round(self.fake_D)

        self.accuracy, self.fake_accuracy, self.real_accuracy, self.loss, self.generator_loss, self.optimizer, \
            self.gradients, self.minimize_loss = self.create_training_method()

        for node in self.gradients:
            try:
                summary = tf.summary.tensor_summary('D_gradients', node[0])
                histogram = tf.summary.histogram('D_gradients', node[0])
                self.var_summary_list.append(summary)
                self.var_summary_list.append(histogram)
            except BaseException:
                print("Wrong summary", node)

        self.accuracy_scalar_summary, self.accuracy_histogram_summary = ops.variable_summaries(self.accuracy,
                                                                                               name='D_acc_summary')
        self.loss_scalar_summary, self.loss_histogram_summary = ops.variable_summaries(self.loss,
                                                                                       name='D_loss_summary')
        # ops.variable_summaries(self.gradients)

    @property
    def var_list(self):
        return self._var_list

    @var_list.setter
    def var_list(self, new_list):
        self._var_list = new_list
        self.gradients = self.optimizer.compute_gradients(loss=self.loss, var_list=self.var_list)

        self.minimize_loss = self.optimizer.minimize(loss=self.loss, var_list=self.var_list)

    def create_model(self, input):

        conv_1 = tf.nn.conv2d(input=input,
                              filter=self.variable_dict['W_1'],
                              strides=[1, self.config.CONV_STRIDE, self.config.CONV_STRIDE, 1],
                              padding="SAME")
        conv_1 = tf.nn.bias_add(conv_1, self.variable_dict['B_1'])

        conv_1 = ops.batch_norm(x=conv_1,
                                beta=self.variable_dict['BETA_1'],
                                gamma=self.variable_dict['GAMMA_1'],
                                phase_train=self.is_training,
                                scope='BATCH_NORM_1')
        conv_1 = ops.leaky_relu(x=conv_1,
                                alpha=0.2,
                                name='LEAKY_RELU_1')

        conv_2 = tf.nn.conv2d(input=conv_1,
                              filter=self.variable_dict['W_2'],
                              strides=[1, self.config.CONV_STRIDE, self.config.CONV_STRIDE, 1],
                              padding="SAME")

        conv_2 = tf.nn.bias_add(conv_2, self.variable_dict['B_2'])

        conv_2 = ops.batch_norm(x=conv_2,
                                beta=self.variable_dict['BETA_2'],
                                gamma=self.variable_dict['GAMMA_2'],
                                phase_train=self.is_training,
                                scope='BATCH_NORM_2')

        conv_2 = ops.leaky_relu(x=conv_2,
                                alpha=0.2,
                                name='LEAKY_RELU_2')

        conv_3 = tf.nn.conv2d(input=conv_2,
                              filter=self.variable_dict['W_3'],
                              strides=[1, self.config.CONV_STRIDE, self.config.CONV_STRIDE, 1],
                              padding="SAME")

        conv_3 = ops.batch_norm(x=conv_3,
                                beta=self.variable_dict['BETA_3'],
                                gamma=self.variable_dict['GAMMA_3'],
                                phase_train=self.is_training,
                                scope='BATCH_NORM_3')

        conv_3 = ops.leaky_relu(x=conv_3,
                                alpha=0.2,
                                name='LEAKY_RELU_3')

        final = tf.reshape(conv_3,
                           [-1, self.config.CONV_OUT_WIDTH * self.config.CONV_OUT_HEIGHT *
                            self.config.CONV_LAYER_3_OUT_CHANNEL])

        final = tf.add(tf.matmul(final, self.variable_dict['W_4']), self.variable_dict['B_4'])

        return tf.nn.sigmoid(final), final

    def create_training_method(self):

        ones_label = tf.ones_like(self.real_D_predication)
        zeros_label = tf.zeros_like(self.fake_D_predication)

        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ones_label,
                                                            logits=self.real_D_logits,
                                                            name='REAL_LOSS')

        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=zeros_label,
                                                            logits=self.fake_D_logits,
                                                            name='FAKE_LOSS')

        g_ones_label = tf.ones_like(self.fake_D_predication)

        generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=g_ones_label,
                                                                                logits=self.fake_D_logits),
                                        name='G_LOSS')

        loss = tf.reduce_mean(real_loss + fake_loss, name='D_LOSS')

        real_accuracy = tf.reduce_mean(tf.cast(x=tf.equal(x=tf.ones_like(self.real_D_predication),
                                                          y=self.real_D_predication),
                                               dtype=tf.float32),
                                       name='REAL_ACCURACY')
        fake_accuracy = tf.reduce_mean(tf.cast(x=tf.equal(x=tf.zeros_like(self.fake_D_predication),
                                                          y=self.fake_D_predication),
                                               dtype=tf.float32),
                                       name='FAKE_ACCURACY')
        accuracy = tf.reduce_mean(fake_accuracy + real_accuracy, name='ACCURACY')

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.LEARNING_RATE)

        if len(self.var_list) > 0:
            gradients = optimizer.compute_gradients(loss=loss, var_list=self.var_list)
            optimize_loss = optimizer.minimize(loss=loss, var_list=self.var_list)
        else:
            gradients = []
            optimize_loss = None

        ops.variable_summaries(fake_accuracy, name='D_acc_summary')
        ops.variable_summaries(real_accuracy, name='D_acc_summary')

        return accuracy, fake_accuracy, real_accuracy, loss, generator_loss, optimizer, gradients, optimize_loss


if __name__ == '__main__':
    with tf.variable_scope('a') as scope:
        a = tf.get_variable(name='a', shape=[1, 2], initializer=tf.constant_initializer(value=0))
        scope.reuse_variables()
        a = tf.get_variable(name='a', shape=[2, 2], initializer=tf.constant_initializer(value=0))
        pass
