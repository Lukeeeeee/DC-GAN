import tensorflow as tf

from src.common.ops import ops
from src.model.deepGAN.Discriminator.discriminatorConfig import DiscriminatorConfig as d_config
from src.model.model import Model


class Discriminator(Model):
    def __init__(self, sess, data, generator):
        super(Discriminator, self).__init__(sess, data)
        self.name = 'Discriminator'

        with tf.variable_scope('Discriminator'):
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
            }

        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=[None, d_config.IN_WIDTH,
                                           d_config.IN_HEIGHT, d_config.IN_CHANNEL],
                                    name='INPUT')

        self.var_list = []
        for key, value in self.variable_dict.iteritems():
            self.var_list.append(value)

        self.is_training = tf.placeholder(tf.bool)

        self.generator = generator

        # self read_D is a size [Batch size * 2] shape tensor
        self.real_D, self.real_D_logits = self.create_model(input=self.input, reuse=False)

        self.real_D_predication = tf.argmax(self.real_D)

        self.fake_D, self.fake_D_logits = self.create_model(input=self.generator, reuse=True)

        self.fake_D_predication = tf.argmax(self.fake_D)

        self.accuracy, self.loss, self.generator_loss, self.optimizer, self.gradients, self.minimize_loss = self.create_training_method()

    def create_model(self, input, reuse):

        # super(Discriminator, self).create_model()

        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            conv_1 = tf.nn.conv2d(input=input,
                                  filter=self.variable_dict['W_1'],
                                  strides=[1, d_config.CONV_STRIDE, d_config.CONV_STRIDE, 1],
                                  padding='SAME')
            conv_1 = tf.nn.bias_add(conv_1, self.variable_dict['B_1'])
            conv_1 = ops.batch_norm(x=conv_1,
                                    beta=self.variable_dict['BETA_1'],
                                    gamma=self.variable_dict['GAMMA_1'],
                                    phase_train=self.is_training,
                                    scope='BATCH_NORM_1')
            conv_1 = ops.leaky_relu(x=conv_1,
                                    alpha=0.2,
                                    name='LEAKY_RELU_1')

            # conv_1 = ops.maxpool2d(x=conv_1, k=2)

            conv_2 = tf.nn.conv2d(input=conv_1,
                                  filter=self.variable_dict['W_2'],
                                  strides=[1, d_config.CONV_STRIDE, d_config.CONV_STRIDE, 1],
                                  padding='same')

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
                                  padding='same')

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
        with tf.variable_scope('Discriminator'):
            ones_label = tf.one_hot(tf.ones_like(self.real_D_predication), depth=2)
            zeros_label = tf.one_hot(tf.zeros_like(self.fake_D_predication), depth=2)

            real_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ones_label,
                                                                logits=self.real_D_logits,
                                                                name='REAL_LOSS')

            fake_loss = tf.nn.softmax_cross_entropy_with_logits(labels=zeros_label,
                                                                logits=self.fake_D_logits,
                                                                name='FAKE_LOSS')

            generator_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ones_label,
                                                                     logits=self.fake_D_logits
                                                                     )

            loss = real_loss + fake_loss

            accuracy = tf.reduce_mean(tf.equal(x=tf.ones_like(self.real_D_predication),
                                               y=self.real_D_predication))
            accuracy = accuracy + tf.reduce_mean(tf.equal(x=tf.zeros_like(self.fake_D_predication),
                                                          y=self.fake_D_predication))

            optimizer = tf.train.RMSPropOptimizer(learning_rate=d_config.LEARNING_RATE)

            gradients = optimizer.compute_gradients(loss=loss, var_list=self.var_list)

            optimize_loss = optimizer.minimize(loss=loss)

        return accuracy, loss, generator_loss, optimizer, gradients, optimize_loss

    def update(self, image_batch, z_batch):
        acc, loss, gradients, _ = self.sess.run(fetches=[self.accuracy, self.loss, self.gradients, self.minimize_loss],
                                                feed_dict={self.input: image_batch, self.generator.input: z_batch})
        return acc, loss, gradients
