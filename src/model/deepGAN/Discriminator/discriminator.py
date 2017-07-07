import tensorflow as tf

from src.common.ops import ops
from src.data.dataConfig import DataConfig
from src.model.deepGAN.Discriminator.discriminatorConfig import DiscriminatorConfig
from src.model.model import Model


class Discriminator(Model):
    def __init__(self, sess, data):
        super(Discriminator, self).__init__(sess, data)
        self.name = 'Discriminator'

        self.variable_dict = {
            "W_1": tf.Variable(tf.truncated_normal([DiscriminatorConfig.FILTER_SIZE, DiscriminatorConfig.FILTER_SIZE,
                                                    DiscriminatorConfig.IN_CHANNEL,
                                                    DiscriminatorConfig.CONV_LAYER_1_OUT_CHANNEL],
                                                   stddev=DiscriminatorConfig.VARIABLE_RANDOM_STANDARD_DEVIATION),
                               name='W_1'),

            "B_1": tf.Variable(tf.constant(0.0, shape=[DiscriminatorConfig.CONV_LAYER_1_OUT_CHANNEL]),
                               name='B_1'),

            'BETA_1': tf.Variable(tf.constant(0.0, shape=[DiscriminatorConfig.CONV_LAYER_1_OUT_CHANNEL]),
                                  name='BETA_1'),

            'GAMMA_1': tf.Variable(tf.random_normal(shape=[DiscriminatorConfig.CONV_LAYER_1_OUT_CHANNEL],
                                                    mean=DiscriminatorConfig.BATCH_NORM_MEAN,
                                                    stddev=DiscriminatorConfig.BATCH_STANDARD_DEVIATION),
                                   name='GAMMA_1'),

            "W_2": tf.Variable(tf.truncated_normal([DiscriminatorConfig.FILTER_SIZE, DiscriminatorConfig.FILTER_SIZE,
                                                    DiscriminatorConfig.CONV_LAYER_1_OUT_CHANNEL,
                                                    DiscriminatorConfig.CONV_LAYER_2_OUT_CHANNEL],
                                                   stddev=DiscriminatorConfig.VARIABLE_RANDOM_STANDARD_DEVIATION),
                               name='W_2'),

            "B_2": tf.Variable(tf.constant(0.0, shape=[DiscriminatorConfig.CONV_LAYER_2_OUT_CHANNEL]),
                               name='B_2'),

            'BETA_2': tf.Variable(tf.constant(0.0, shape=[DiscriminatorConfig.CONV_LAYER_2_OUT_CHANNEL]),
                                  name='BETA_2'),

            'GAMMA_2': tf.Variable(tf.random_normal(shape=[DiscriminatorConfig.CONV_LAYER_2_OUT_CHANNEL],
                                                    mean=DiscriminatorConfig.BATCH_NORM_MEAN,
                                                    stddev=DiscriminatorConfig.BATCH_STANDARD_DEVIATION),
                                   name='GAMMA_2'),

            "W_3": tf.Variable(tf.truncated_normal([DiscriminatorConfig.FILTER_SIZE, DiscriminatorConfig.FILTER_SIZE,
                                                    DiscriminatorConfig.CONV_LAYER_2_OUT_CHANNEL,
                                                    DiscriminatorConfig.CONV_LAYER_3_OUT_CHANNEL],
                                                   stddev=DiscriminatorConfig.VARIABLE_RANDOM_STANDARD_DEVIATION),
                               name='W_3'),

            "B_3": tf.Variable(tf.constant(0.0, shape=[DiscriminatorConfig.CONV_LAYER_3_OUT_CHANNEL]),
                               name='B_3'),

            'BETA_3': tf.Variable(tf.constant(0.0, shape=[DiscriminatorConfig.CONV_LAYER_3_OUT_CHANNEL]),
                                  name='BETA_3'),

            'GAMMA_3': tf.Variable(tf.random_normal(shape=[DiscriminatorConfig.CONV_LAYER_3_OUT_CHANNEL],
                                                    mean=DiscriminatorConfig.BATCH_NORM_MEAN,

                                                    stddev=DiscriminatorConfig.BATCH_STANDARD_DEVIATION),
                                   name='GAMMA_3'),

            "W_4": tf.Variable(tf.truncated_normal([(DiscriminatorConfig.CONV_OUT_HEIGHT *
                                                     DiscriminatorConfig.CONV_OUT_WIDTH *
                                                     DiscriminatorConfig.CONV_LAYER_3_OUT_CHANNEL),
                                                    DiscriminatorConfig.OUTPUT_SIZE],
                                                   stddev=DiscriminatorConfig.VARIABLE_RANDOM_STANDARD_DEVIATION),
                               name='W_4'),

            "B_4": tf.Variable(tf.constant(0.0, shape=[DiscriminatorConfig.OUTPUT_SIZE]), name='b_4')
        }

        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=[DataConfig.BATCH_SIZE, DiscriminatorConfig.IN_WIDTH,
                                           DiscriminatorConfig.IN_HEIGHT, DiscriminatorConfig.IN_CHANNEL])
        self.label = tf.placeholder(dtype=tf.float32,
                                    shape=[DataConfig.BATCH_SIZE, DiscriminatorConfig.OUTPUT_SIZE])

        self.var_list = []
        for key, value in self.variable_dict.iteritems():
            self.var_list.append(value)

        self.is_training = tf.placeholder(tf.bool)

        self.predication = self.create_model()

        self.accuracy, self.loss, self.optimizer, self.gradients, self.minimize_loss = self.create_training_method()

    def create_model(self):
        with tf.variable_scope('Discriminator'):
            conv_1 = tf.nn.conv2d(input=self.input,
                                  filter=self.variable_dict['W_1'],
                                  strides=[1, DiscriminatorConfig.CONV_STRIDE, DiscriminatorConfig.CONV_STRIDE, 1],
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
                                  strides=[1, DiscriminatorConfig.CONV_STRIDE, DiscriminatorConfig.CONV_STRIDE, 1],
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
                                  strides=[1, DiscriminatorConfig.CONV_STRIDE, DiscriminatorConfig.CONV_STRIDE, 1],
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
                               [-1, DiscriminatorConfig.CONV_OUT_WIDTH * DiscriminatorConfig.CONV_OUT_HEIGHT *
                                DiscriminatorConfig.CONV_LAYER_3_OUT_CHANNEL])

            final = tf.add(tf.matmul(final, self.variable_dict['W_4']), self.variable_dict['B_4'])

            return final

    def create_training_method(self):
        with tf.variable_scope('Discriminator'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label,
                                                                  logits=self.predication,
                                                                  name='LOSS')
            accuracy = tf.reduce_mean(tf.equal(self.label, tf.argmax(tf.nn.softmax(self.predication))))

            optimizer = tf.train.RMSPropOptimizer(learning_rate=DiscriminatorConfig.LEARNING_RATE)

            gradients = optimizer.compute_gradients(loss=loss, var_list=self.var_list)

            optimize_loss = optimizer.minimize(loss=loss)

        return accuracy, loss, optimizer, gradients, optimize_loss

    def predicate(self, data):
        loss, pred, acc = self.sess.run(fetches=[self.minimize_loss, self.predication, self.accuracy],
                                        feed_dict={self.input: data['input'], self.label: data['label']})
        return pred, loss, acc
