import tensorflow as tf

from src.common.ops import ops
from src.model.deepGAN.Discriminator.discriminatorConfig import DiscriminatorConfig as d_config
from src.model.model import Model


class Discriminator(Model):
    def __init__(self, sess, data, generator):
        super(Discriminator, self).__init__(sess, data)
        self.name = 'Discriminator'
        with tf.variable_scope(self.name):
            self.variable_dict = {
                "W_1": tf.Variable(tf.truncated_normal([d_config.FILTER_SIZE, d_config.FILTER_SIZE,
                                                        d_config.IN_CHANNEL,
                                                        d_config.CONV_LAYER_1_OUT_CHANNEL],
                                                       stddev=d_config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                                   name='W_1'),


                "B_1": tf.Variable(tf.constant(0.0, shape=[d_config.CONV_LAYER_1_OUT_CHANNEL]),
                                   name='B_1'),

                'BETA_1': tf.Variable(tf.constant(0.0, shape=[d_config.CONV_LAYER_1_OUT_CHANNEL]),
                                      name='BETA_1'),

                'GAMMA_1': tf.Variable(tf.random_normal(shape=[d_config.CONV_LAYER_1_OUT_CHANNEL],
                                                        mean=d_config.BATCH_NORM_MEAN,
                                                        stddev=d_config.BATCH_STANDARD_DEVIATION),
                                       name='GAMMA_1'),

                "W_2": tf.Variable(tf.truncated_normal([d_config.FILTER_SIZE, d_config.FILTER_SIZE,
                                                        d_config.CONV_LAYER_1_OUT_CHANNEL,
                                                        d_config.CONV_LAYER_2_OUT_CHANNEL],
                                                       stddev=d_config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                                   name='W_2'),

                "B_2": tf.Variable(tf.constant(0.0, shape=[d_config.CONV_LAYER_2_OUT_CHANNEL]),
                                   name='B_2'),

                'BETA_2': tf.Variable(tf.constant(0.0, shape=[d_config.CONV_LAYER_2_OUT_CHANNEL]),
                                      name='BETA_2'),

                'GAMMA_2': tf.Variable(tf.random_normal(shape=[d_config.CONV_LAYER_2_OUT_CHANNEL],
                                                        mean=d_config.BATCH_NORM_MEAN,
                                                        stddev=d_config.BATCH_STANDARD_DEVIATION),
                                       name='GAMMA_2'),

                "W_3": tf.Variable(tf.truncated_normal([d_config.FILTER_SIZE, d_config.FILTER_SIZE,
                                                        d_config.CONV_LAYER_2_OUT_CHANNEL,
                                                        d_config.CONV_LAYER_3_OUT_CHANNEL],
                                                       stddev=d_config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                                   name='W_3'),

                "B_3": tf.Variable(tf.constant(0.0, shape=[d_config.CONV_LAYER_3_OUT_CHANNEL]),
                                   name='B_3'),

                'BETA_3': tf.Variable(tf.constant(0.0, shape=[d_config.CONV_LAYER_3_OUT_CHANNEL]),
                                      name='BETA_3'),

                'GAMMA_3': tf.Variable(tf.random_normal(shape=[d_config.CONV_LAYER_3_OUT_CHANNEL],
                                                        mean=d_config.BATCH_NORM_MEAN,

                                                        stddev=d_config.BATCH_STANDARD_DEVIATION),
                                       name='GAMMA_3'),

                "W_4": tf.Variable(tf.truncated_normal([(d_config.CONV_OUT_HEIGHT *
                                                         d_config.CONV_OUT_WIDTH *
                                                         d_config.CONV_LAYER_3_OUT_CHANNEL),
                                                        d_config.OUTPUT_SIZE],
                                                       stddev=d_config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                                   name='W_4'),

                "B_4": tf.Variable(tf.constant(0.0, shape=[d_config.OUTPUT_SIZE]), name='b_4')

                # "W_1": tf.get_variable(shape=[d_config.FILTER_SIZE, d_config.FILTER_SIZE,
                #                               d_config.IN_CHANNEL,
                #                               d_config.CONV_LAYER_1_OUT_CHANNEL],
                #                        initializer=tf.truncated_normal_initializer(
                #                            stddev=d_config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                #                        name='W_1'),
                #
                # "B_1": tf.get_variable(shape=[d_config.CONV_LAYER_1_OUT_CHANNEL],
                #                        name='B_1',
                #                        initializer=tf.constant_initializer(value=0.0)),
                #
                # 'BETA_1': tf.get_variable(shape=[d_config.CONV_LAYER_1_OUT_CHANNEL],
                #                           name='BETA_1',
                #                           initializer=tf.constant_initializer(value=0.0)),
                #
                # 'GAMMA_1': tf.get_variable(shape=[d_config.CONV_LAYER_1_OUT_CHANNEL],
                #                            initializer=tf.random_normal_initializer(mean=d_config.BATCH_NORM_MEAN,
                #                                                                     stddev=d_config.BATCH_STANDARD_DEVIATION),
                #                            name='GAMMA_1'),
                #
                # 'W_2': tf.get_variable(shape=[d_config.FILTER_SIZE, d_config.FILTER_SIZE,
                #                               d_config.CONV_LAYER_1_OUT_CHANNEL,
                #                               d_config.CONV_LAYER_2_OUT_CHANNEL],
                #                        initializer=tf.truncated_normal_initializer(
                #                            stddev=d_config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                #                        name='W_2'),
                #
                # 'B_2': tf.get_variable(shape=[d_config.CONV_LAYER_2_OUT_CHANNEL],
                #                        initializer=tf.constant_initializer(value=0.0),
                #                        name='B_2'),
                #
                # 'BETA_2': tf.get_variable(shape=[d_config.CONV_LAYER_2_OUT_CHANNEL],
                #                           initializer=tf.constant_initializer(value=0.0),
                #                           name='BETA_2'),
                #
                # 'GAMMA_2': tf.get_variable(shape=[d_config.CONV_LAYER_2_OUT_CHANNEL],
                #                            initializer=tf.random_normal_initializer(mean=d_config.BATCH_NORM_MEAN,
                #                                                                     stddev=d_config.BATCH_STANDARD_DEVIATION),
                #                            name='GAMMA_2'),
                #
                # 'W_3': tf.get_variable(shape=[d_config.FILTER_SIZE, d_config.FILTER_SIZE,
                #                               d_config.CONV_LAYER_2_OUT_CHANNEL,
                #                               d_config.CONV_LAYER_3_OUT_CHANNEL],
                #                        initializer=tf.truncated_normal_initializer(
                #                            stddev=d_config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                #                        name='W_3'),
                #
                # 'B_3': tf.get_variable(shape=[d_config.CONV_LAYER_3_OUT_CHANNEL],
                #                        initializer=tf.constant_initializer(value=0.0),
                #                        name='B_3'),
                #
                # 'BETA_3': tf.get_variable(shape=[d_config.CONV_LAYER_3_OUT_CHANNEL],
                #                           initializer=tf.constant_initializer(value=0.0),
                #                           name='BETA_3'),
                #
                # 'GAMMA_3': tf.get_variable(shape=[d_config.CONV_LAYER_3_OUT_CHANNEL],
                #                            initializer=tf.random_normal_initializer(mean=d_config.BATCH_NORM_MEAN,
                #                                                                     stddev=d_config.BATCH_STANDARD_DEVIATION),
                #                            name='GAMMA_3'),
                #
                # 'W_4': tf.get_variable(shape=[(d_config.CONV_OUT_HEIGHT * d_config.CONV_OUT_WIDTH
                #                                * d_config.CONV_LAYER_3_OUT_CHANNEL), d_config.OUTPUT_SIZE],
                #                        initializer=tf.truncated_normal_initializer(
                #                        stddev=d_config.VARIABLE_RANDOM_STANDARD_DEVIATION),
                #                        name='W_4'),
                #
                # 'B_4': tf.get_variable(shape=[d_config.OUTPUT_SIZE],
                #                        initializer=tf.constant_initializer(value=0.0),
                #                        name='B_4'),

            }

            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=[None, d_config.IN_WIDTH,
                                               d_config.IN_HEIGHT, d_config.IN_CHANNEL],
                                        name='D_INPUT')
            self.is_training = tf.placeholder(tf.bool)

        self.var_list = []
        for key, value in self.variable_dict.iteritems():
            self.var_list.append(value)

        self.generator = generator

        # self read_D is a size [Batch size * 2] shape tensor
        self.real_D, self.real_D_logits = self.create_model(input=self.input)

        self.real_D_predication = tf.argmax(self.real_D, axis=1)

        self.fake_D, self.fake_D_logits = self.create_model(input=self.generator.output)

        self.fake_D_predication = tf.argmax(self.fake_D, axis=1)

        self.accuracy, self.fake_accuracy, self.real_accuracy, self.loss, self.generator_loss, self.optimizer, \
        self.gradients, self.minimize_loss = self.create_training_method()

        self.accuracy_scalar_summary, self.accuracy_histogram_summary = ops.variable_summaries(self.accuracy)
        self.loss_scalar_summary, self.loss_histogram_summary = ops.variable_summaries(self.loss)
        # ops.variable_summaries(self.gradients)

    def create_model(self, input):

        # super(Discriminator, self).create_model()

        with tf.variable_scope(self.name):

            conv_1 = tf.nn.conv2d(input=input,
                                  filter=self.variable_dict['W_1'],
                                  strides=[1, d_config.CONV_STRIDE, d_config.CONV_STRIDE, 1],
                                  padding="SAME")
            conv_1 = tf.nn.bias_add(conv_1, self.variable_dict['B_1'])

            # conv_1 = tf.layers.batch_normalization(inputs=conv_1, reuse=None, name='BATCH_NORM_1')

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
                                  strides=[1, d_config.CONV_STRIDE, d_config.CONV_STRIDE, 1],
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
                                  strides=[1, d_config.CONV_STRIDE, d_config.CONV_STRIDE, 1],
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
                               [-1, d_config.CONV_OUT_WIDTH * d_config.CONV_OUT_HEIGHT *
                                d_config.CONV_LAYER_3_OUT_CHANNEL])

            final = tf.add(tf.matmul(final, self.variable_dict['W_4']), self.variable_dict['B_4'])

            return tf.nn.softmax(final), final

    def create_training_method(self):
        with tf.variable_scope(self.name):

            ones_label = tf.one_hot(tf.ones_like(self.real_D_predication), depth=2)
            zeros_label = tf.one_hot(tf.zeros_like(self.fake_D_predication), depth=2)

            real_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ones_label,
                                                                logits=self.real_D_logits,
                                                                name='REAL_LOSS')

            fake_loss = tf.nn.softmax_cross_entropy_with_logits(labels=zeros_label,
                                                                logits=self.fake_D_logits,
                                                                name='FAKE_LOSS')

            g_ones_label = tf.one_hot(tf.ones_like(self.fake_D_predication), depth=2)

            generator_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=g_ones_label,
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
            accuracy = tf.div(tf.add(fake_accuracy, real_accuracy), tf.constant(2.0), name='ACCURACY')

            optimizer = tf.train.RMSPropOptimizer(learning_rate=d_config.LEARNING_RATE)

            gradients = optimizer.compute_gradients(loss=loss, var_list=self.var_list)

            optimize_loss = optimizer.minimize(loss=loss)

            self.fake_accuracy_scalar_summary, self.fake_accuracy_histogram_summary = ops.variable_summaries(
                fake_accuracy)
            self.real_accuracy_scalar_summary, self.real_accuracy_histogram_summary = ops.variable_summaries(
                real_accuracy)

        return accuracy, fake_accuracy, real_accuracy, loss, generator_loss, optimizer, gradients, optimize_loss


if __name__ == '__main__':
    with tf.variable_scope('a') as scope:
        a = tf.get_variable(name='a', shape=[1, 2], initializer=tf.constant_initializer(value=0))
        scope.reuse_variables()
        a = tf.get_variable(name='a', shape=[2, 2], initializer=tf.constant_initializer(value=0))
        pass
