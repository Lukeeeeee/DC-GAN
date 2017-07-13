import tensorflow as tf


class ops(object):
    @staticmethod
    def leaky_relu(x, alpha=0.1, name='lrelu'):
        with tf.variable_scope(name):
            return tf.maximum(tf.multiply(alpha, x, name=name + 'lrelu/add'), x, name=name + 'lrelu/maxmium')

    @staticmethod
    def batch_norm(x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):
        # TODO CHANGE BATCH NORM
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            # beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), trainable=True)
            # gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev), trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
            # normed = tf.layers.batch_normalization(x)
            return normed

    @staticmethod
    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    @staticmethod
    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('Summary'):
            scalar = tf.summary.scalar(var.name, var)

            # tf.summary.scalar('mean', mean)
            # tf.summary.scalar('max', tf.reduce_max(var))
            # tf.summary.scalar('min', tf.reduce_min(var))
            histogram = tf.summary.histogram(var.name, var)
            return scalar, histogram


# try:
#   image_summary = tf.image_summary
#   scalar_summary = tf.scalar_summary
#   histogram_summary = tf.histogram_summary
#   merge_summary = tf.merge_summary
#   SummaryWriter = tf.train.SummaryWriter
# except:
#   image_summary = tf.summary.image
#   scalar_summary = tf.summary.scalar
#   histogram_summary = tf.summary.histogram
#   merge_summary = tf.summary.merge
#   SummaryWriter = tf.summary.FileWriter
#
# if "concat_v2" in dir(tf):
#   def concat(tensors, axis, *args, **kwargs):
#     return tf.concat_v2(tensors, axis, *args, **kwargs)
# else:
#   def concat(tensors, axis, *args, **kwargs):
#     return tf.concat(tensors, axis, *args, **kwargs)
#
# class batch_norm(object):
#   def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
#     with tf.variable_scope(name):
#       self.epsilon  = epsilon
#       self.momentum = momentum
#       self.name = name
#
#   def __call__(self, x, train=True):
#     return tf.contrib.layers.batch_norm(x,
#                       decay=self.momentum,
#                       updates_collections=None,
#                       epsilon=self.epsilon,
#                       scale=True,
#                       is_training=train,
#                       scope=self.name)
#
# def conv_cond_concat(x, y):
#   """Concatenate conditioning vector on feature map axis."""
#   x_shapes = x.get_shape()
#   y_shapes = y.get_shape()
#   return concat([
#     x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)
#
# def conv2d(input_, output_dim,
#        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#        name="conv2d"):
#   with tf.variable_scope(name):
#     w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
#               initializer=tf.truncated_normal_initializer(stddev=stddev))
#     conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
#
#     biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
#     conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
#
#     return conv
#
# def deconv2d(input_, output_shape,
#        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#        name="deconv2d", with_w=False):
#   with tf.variable_scope(name):
#     # filter : [height, width, output_channels, in_channels]
#     w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
#               initializer=tf.random_normal_initializer(stddev=stddev))
#
#     try:
#       deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
#                 strides=[1, d_h, d_w, 1])
#
#     # Support for verisons of TensorFlow before 0.7.0
#     except AttributeError:
#       deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
#                 strides=[1, d_h, d_w, 1])
#
#     biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
#     deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
#
#     if with_w:
#       return deconv, w, biases
#     else:
#       return deconv
#
# def lrelu(x, leak=0.2, name="lrelu"):
#   return tf.maximum(x, leak*x)
#
# def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
#   shape = input_.get_shape().as_list()
#
#   with tf.variable_scope(scope or "Linear"):
#     matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
#                  tf.random_normal_initializer(stddev=stddev))
#     bias = tf.get_variable("bias", [output_size],
#       initializer=tf.constant_initializer(bias_start))
#     if with_w:
#       return tf.matmul(input_, matrix) + bias, matrix, bias
#     else:
#       return tf.matmul(input_, matrix) + bias
