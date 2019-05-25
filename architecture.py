import tensorflow as tf
import tensorflow.layers as tfl

from tensorflow.contrib.distributions import MultivariateNormalDiag

# VARIOUS ARCHITECTURES
class RectangleCoder():
    def __init__(self, dataset,
            h_dim=256,
            kernel_size=3,
            kernel_num=32):

        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.image_shape = dataset.image_shape

    def encode(self, x, z_dim):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            h = tf.reshape(h, (-1, im_h, im_w, im_c))

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same")
            h = tfl.max_pooling2d(h, 2, 2)
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same")
            h = tfl.max_pooling2d(h, 2, 2)
            h = tf.nn.relu(h)


            h = tfl.flatten(h)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            return z_mean

    def decode(self, z, x_dim):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(
                    h, units=im_h // 4 * im_w // 4 * self.kernel_num,
                    activation=tf.nn.relu)
            h = tf.reshape(h, (-1, im_h // 4, im_w // 4, self.kernel_num))

            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d_transpose(
                    h, im_c, self.kernel_size,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same")

            h = tfl.flatten(h)
            y_mean = h
            return y_mean

class CelebaCoder():
    def __init__(self, dataset,
            h_dim=256,
            kernel_size=4,
            kernel_num=32):

        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.image_shape = dataset.image_shape

    def encode(self, x, z_dim, training=False):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            h = tf.reshape(h, (-1, im_h, im_w, im_c))

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            h = tfl.flatten(h)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            # z_mean = tfl.batch_normalization(z_mean, training=training)
            return z_mean

    def decode(self, z, x_dim, training=False):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            stride = 16
            h = tfl.dense(
                    h, units=im_h // stride * im_w // stride * self.kernel_num * 2,
                    activation=tf.nn.relu)
            h = tf.reshape(h, (-1, im_h // stride, im_w // stride, self.kernel_num * 2))


            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=2, padding="same", activation=tf.nn.relu)

            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same", activation=tf.nn.relu)

            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same", activation=tf.nn.relu)

            h = tfl.conv2d_transpose(
                    h, im_c, self.kernel_size,
                    strides=2, padding="same", activation=tf.nn.sigmoid)

            h = tfl.flatten(h)
            y_mean = h
            return y_mean

class CifarCoder():
    def __init__(self, dataset,
            h_dim=256,
            kernel_size=3,
            kernel_num=32):

        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.image_shape = dataset.image_shape

    def encode(self, x, z_dim, training=False):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            h = tf.reshape(h, (-1, im_h, im_w, im_c))
            # TODO:
            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            # TODO: tu troche zmienilem
            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            h = tfl.flatten(h)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            # z_mean = tfl.batch_normalization(z_mean, training=training)
            return z_mean

    def decode(self, z, x_dim, training=False):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            # h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            stride = 8
            h = tfl.dense(
                    h, units=im_h // stride * im_w // stride * self.kernel_num * 2,
                    activation=tf.nn.relu)
            h = tf.reshape(h, (-1, im_h // stride, im_w // stride, self.kernel_num * 2))

            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=2, padding="same", activation=tf.nn.relu)

            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same", activation=tf.nn.relu)

            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same", activation=tf.nn.relu)

            h = tfl.conv2d_transpose(
                    h, im_c, self.kernel_size,
                    strides=1, padding="same", activation=tf.nn.sigmoid)

            h = tfl.flatten(h)
            y_mean = h
            return y_mean

class ReversePyramidCoder():
    def __init__(self, dataset,
            h_dim=0,
            kernel_size=2,
            kernel_num=128):

        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.image_shape = dataset.image_shape

    def encode(self, x, z_dim):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            h = tf.reshape(h, (-1, im_h, im_w, im_c))
            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    padding="valid", strides=2, activation=tf.nn.relu)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size,
                    padding="valid", strides=2, activation=tf.nn.relu)

            h = tfl.conv2d(
                    h, self.kernel_num * 3, self.kernel_size,
                    padding="valid", strides=2, activation=tf.nn.relu)


            h = tfl.flatten(h)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            return z_mean

    def decode(self, z, x_dim):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(
                    h, units=im_h // 8 * im_w // 8 * self.kernel_num,
                    activation=tf.nn.relu)
            h = tf.reshape(h, (-1, im_h // 8, im_w // 8, self.kernel_num))

            # h = tfl.conv2d_transpose(
            #         h, self.kernel_num, self.kernel_size - 2,
            #         strides=1, padding="valid", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 3, self.kernel_size,
                    strides=2, padding="valid", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=2, padding="valid", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 1, self.kernel_size,
                    strides=2, padding="valid", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, im_c, 1,
                    strides=1, padding="valid", activation=tf.nn.relu)

            h = tfl.flatten(h)
            y_mean = h
            return y_mean

class WideShaoClassifierCoder():
    def __init__(self, dataset,
            h_dim=256,
            kernel_size=4,
            kernel_num=32):

        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.image_shape = dataset.image_shape

    def encode(self, x, y, z_dim, training=False):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            h = tf.reshape(h, (-1, im_h, im_w, im_c))
            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=1, padding="same")
            h = tfl.max_pooling2d(h, 2, 2)
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=1, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same")
            h = tfl.max_pooling2d(h, 2, 2)
            h = tf.nn.relu(h)

            h = tfl.flatten(h)
            y = tfl.dense(h, units=self.h_dim // 4, activation=tf.nn.relu)
            h = tf.concat([h, y], 1)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            # h = tfl.dense(h, units=self.h_dim // 2, activation=tf.nn.relu)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            return z_mean

    def decode(self, z, x_dim, training=False):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            # h = tfl.dense(h, units=self.h_dim // 2, activation=tf.nn.relu)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(
                    h, units=im_h // 4 * im_w // 4 * self.kernel_num,
                    activation=tf.nn.relu)
            h = tf.reshape(h, (-1, im_h // 4, im_w // 4, self.kernel_num))

            # h = tfl.conv2d_transpose(
            #         h, self.kernel_num, self.kernel_size - 2,
            #         strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=1, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d_transpose(
                    h, im_c, self.kernel_size,
                    strides=1, padding="same")

            h = tfl.flatten(h)
            y_mean = h
            return y_mean

class WideShaoCoder():
    def __init__(self, dataset,
            h_dim=256,
            kernel_size=4,
            kernel_num=32):

        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.image_shape = dataset.image_shape

    def encode(self, x, z_dim, training=False):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            h = tf.reshape(h, (-1, im_h, im_w, im_c))
            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=1, padding="same")
            h = tfl.max_pooling2d(h, 2, 2)
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=1, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same")
            h = tfl.max_pooling2d(h, 2, 2)
            h = tf.nn.relu(h)

            h = tfl.flatten(h)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            # h = tfl.dense(h, units=self.h_dim // 2, activation=tf.nn.relu)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            return z_mean

    def decode(self, z, x_dim, training=False):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            # h = tfl.dense(h, units=self.h_dim // 2, activation=tf.nn.relu)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(
                    h, units=im_h // 4 * im_w // 4 * self.kernel_num,
                    activation=tf.nn.relu)
            h = tf.reshape(h, (-1, im_h // 4, im_w // 4, self.kernel_num))

            # h = tfl.conv2d_transpose(
            #         h, self.kernel_num, self.kernel_size - 2,
            #         strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=1, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d_transpose(
                    h, im_c, self.kernel_size,
                    strides=1, padding="same")

            h = tfl.flatten(h)
            y_mean = h
            return y_mean

class SimpleCnnCoder():
    def __init__(self, dataset,
            h_dim=300,
            kernel_size=4,
            kernel_num=32):

        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.image_shape = dataset.image_shape

    def encode(self, x, z_dim, training):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            h = tf.reshape(h, (-1, im_h, im_w, im_c))

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.max_pooling2d(h, 2, 2)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.max_pooling2d(h, 2, 2)

            h = tfl.flatten(h)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            return z_mean

    def decode(self, z, x_dim, training):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(
                    h, units=im_h // 4 * im_w // 4 * self.kernel_num * 2,
                    activation=tf.nn.relu)
            h = tf.reshape(h, (-1, im_h // 4, im_w // 4, self.kernel_num * 2))

            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same", activation=tf.nn.relu)

            h = tfl.conv2d_transpose(
                    h, im_c, self.kernel_size,
                    strides=2, padding="same")

            h = tfl.flatten(h)
            y_mean = h
            return y_mean

class ShaoCoder():
    def __init__(self, dataset,
            h_dim=256,
            kernel_size=5,
            kernel_num=32):

        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.image_shape = dataset.image_shape

    def encode(self, x, z_dim):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            h = tf.reshape(h, (-1, im_h, im_w, im_c))

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same")
            h = tfl.max_pooling2d(h, 2, 2)
            h = tf.nn.relu(h)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size - 1,
                    strides=1, padding="same")
            h = tfl.max_pooling2d(h, 2, 2)
            h = tf.nn.relu(h)

            h = tfl.flatten(h)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            # h = tfl.dense(h, units=self.h_dim // 2, activation=tf.nn.relu)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            return z_mean

    def decode(self, z, x_dim):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            # h = tfl.dense(h, units=self.h_dim // 2, activation=tf.nn.relu)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(
                    h, units=im_h // 4 * im_w // 4 * self.kernel_num * 2,
                    activation=tf.nn.relu)
            h = tf.reshape(h, (-1, im_h // 4, im_w // 4, self.kernel_num * 2))

            # h = tfl.conv2d_transpose(
            #         h, self.kernel_num, self.kernel_size - 2,
            #         strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, self.kernel_size - 1,
                    strides=2, padding="same")
            h = tf.nn.relu(h)

            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same")
            # h = tf.nn.relu(h)

            h = tfl.flatten(h)
            y_mean = h
            return y_mean

class VaeRectSvhnCoder():
    def __init__(self, dataset,
            h_dim=400,
            kernel_size=3,
            kernel_num=25):

        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.image_shape = dataset.image_shape

    def encode(self, x, z_dim, training):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            h = tf.reshape(h, (-1, im_h, im_w, im_c))

            # BLOCK 1
            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    padding="same", activation=tf.nn.relu)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    padding="same", activation=tf.nn.relu)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    padding="same", activation=tf.nn.relu)
            h = tfl.max_pooling2d(h, 2, 2)


            # BLOCK 3
            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size,
                    padding="same", activation=tf.nn.relu)
            h = tfl.conv2d(
                    h, self.kernel_num * 2, 1,
                    padding="same", activation=tf.nn.relu)
            h = tfl.conv2d(
                    h, self.kernel_num * 2, 1,
                    padding="same", activation=tf.nn.relu)
            h = tfl.max_pooling2d(h, 2, 2)

            h = tfl.flatten(h)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            z_var = tfl.dense(h, units=z_dim, name='z_mean', activation=tf.nn.softplus)

            distr = MultivariateNormalDiag(loc=z_mean, scale_diag=z_var)
            # z_mean = tfl.batch_normalization(z_mean, training=training)
            return distr, distr.sample()

    def decode(self, z, x_dim, training):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(
                    h, units=im_h // 4 * im_w // 4 * self.kernel_num * 2,
                    activation=tf.nn.relu)
            h = tf.reshape(h, (-1, im_h // 4, im_w // 4, self.kernel_num * 2))

            # BLOCK 3
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, 1,
                    strides=2, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, 1,
                    strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=1, padding="same", activation=tf.nn.relu)

            # BLOCK 1
            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, im_c, self.kernel_size,
                    strides=1, padding="same", activation=tf.nn.sigmoid)

            h = tfl.flatten(h)
            y_mean = h
            return y_mean



class RectSvhnCoder():
    def __init__(self, dataset,
            h_dim=400,
            kernel_size=3,
            kernel_num=25):

        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.image_shape = dataset.image_shape

    def encode(self, x, z_dim, training):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            h = tf.reshape(h, (-1, im_h, im_w, im_c))

            # BLOCK 1
            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    padding="same", activation=tf.nn.relu)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    padding="same", activation=tf.nn.relu)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    padding="same", activation=tf.nn.relu)
            h = tfl.max_pooling2d(h, 2, 2)

            # BLOCK 2
            # h = tfl.conv2d(
            #         h, self.kernel_num * 2, self.kernel_size,
            #         padding="same", activation=tf.nn.relu)

            # h = tfl.conv2d(
            #         h, self.kernel_num * 2, self.kernel_size,
            #         padding="same", activation=tf.nn.relu)

            # h = tfl.conv2d(
            #         h, self.kernel_num * 2, self.kernel_size,
            #         padding="same", activation=tf.nn.relu)

            # BLOCK 3
            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size,
                    padding="same", activation=tf.nn.relu)
            h = tfl.conv2d(
                    h, self.kernel_num * 2, 1,
                    padding="same", activation=tf.nn.relu)
            h = tfl.conv2d(
                    h, self.kernel_num * 2, 1,
                    padding="same", activation=tf.nn.relu)
            h = tfl.max_pooling2d(h, 2, 2)

            h = tfl.flatten(h)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            # z_mean = tfl.batch_normalization(z_mean, training=training)
            return z_mean

    def decode(self, z, x_dim, training):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(
                    h, units=im_h // 4 * im_w // 4 * self.kernel_num * 2,
                    activation=tf.nn.relu)
            h = tf.reshape(h, (-1, im_h // 4, im_w // 4, self.kernel_num * 2))

            # BLOCK 3
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, 1,
                    strides=2, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, 1,
                    strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, self.kernel_size,
                    strides=1, padding="same", activation=tf.nn.relu)

            # BLOCK 2
            # h = tfl.conv2d_transpose(
            #         h, self.kernel_num * 2, self.kernel_size,
            #         strides=2, padding="same", activation=tf.nn.relu)
            # h = tfl.conv2d_transpose(
            #         h, self.kernel_num * 2, self.kernel_size,
            #         strides=1, padding="same", activation=tf.nn.relu)
            # h = tfl.conv2d_transpose(
            #         h, self.kernel_num, self.kernel_size,
            #         strides=1, padding="same", activation=tf.nn.relu)

            # BLOCK 3
            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, im_c, self.kernel_size,
                    strides=1, padding="same", activation=tf.nn.sigmoid)

            h = tfl.flatten(h)
            y_mean = h
            return y_mean

class RectLadderSmallCoder():
    def __init__(self, dataset,
            h_dim=400,
            kernel_size=5,
            kernel_num=25):

        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.image_shape = dataset.image_shape

    def encode(self, x, z_dim, training):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            h = tf.reshape(h, (-1, im_h, im_w, im_c))
            h = tfl.conv2d(
                    h, self.kernel_num, 5,
                    padding="same", activation=tf.nn.relu)

            h = tfl.max_pooling2d(h, 2, 2)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, 3,
                    padding="same", activation=tf.nn.relu)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, 3,
                    padding="same", activation=tf.nn.relu)

            h = tfl.max_pooling2d(h, 2, 2)
            h = tfl.conv2d(
                    h, self.kernel_num * 4, 3,
                    padding="same", activation=tf.nn.relu)

            h = tfl.conv2d(
                    h, self.h_dim, 1,
                    padding="same", activation=tf.nn.relu)

            h = tfl.flatten(h)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            # h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            return z_mean

    def decode(self, z, x_dim, training):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            # h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(
                    h, units=im_h // 4 * im_w // 4 * self.h_dim,
                    activation=tf.nn.relu)
            h = tf.reshape(h, (-1, im_h // 4, im_w // 4, self.h_dim))

            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 4, 1,
                    strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, 3,
                    strides=2, padding="same", activation=tf.nn.relu)

            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, 3,
                    strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num, 3,
                    strides=2, padding="same", activation=tf.nn.relu)

            h = tfl.conv2d_transpose(
                    h, im_c, 5,
                    strides=1, padding="same", activation=None)

            h = tfl.flatten(h)
            y_mean = h
            return y_mean

class RectCnnCoder():
    def __init__(self, dataset,
            h_dim=400,
            kernel_size=5,
            kernel_num=25):

        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.image_shape = dataset.image_shape

    def encode(self, x, z_dim, training):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            h = tf.reshape(h, (-1, im_h, im_w, im_c))
            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    padding="same", activation=tf.nn.relu)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    padding="same", activation=tf.nn.relu)
            h = tfl.max_pooling2d(h, 2, 2)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    padding="same", activation=tf.nn.relu)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    padding="same", activation=tf.nn.relu)
            h = tfl.max_pooling2d(h, 2, 2)
            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size,
                    padding="same", activation=tf.nn.relu)

            h = tfl.flatten(h)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            # h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            return z_mean

    def decode(self, z, x_dim, training):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            # h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(
                    h, units=im_h // 4 * im_w // 4 * self.kernel_num,
                    activation=tf.nn.relu)
            h = tf.reshape(h, (-1, im_h // 4, im_w // 4, self.kernel_num))

            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num, self.kernel_size,
                    strides=2, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, im_c, self.kernel_size,
                    strides=1, padding="same", activation=None)

            h = tfl.flatten(h)
            y_mean = h
            return y_mean


class CnnCoder():
    def __init__(
            self, dataset,
            h_dim=400,
            kernel_size=5,
            kernel_num=25):

        self.h_dim = h_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.image_shape = dataset.image_shape

    def encode(self, x, z_dim):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            h = tf.reshape(h, (-1, im_h, im_w, im_c))
            h = tfl.conv2d(
                    h, self.kernel_num * 4, self.kernel_size,
                    padding="same", activation=tf.nn.relu)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size,
                    padding="same", activation=tf.nn.relu)
            h = tfl.max_pooling2d(h, 2, 2)

            h = tfl.conv2d(
                    h, self.kernel_num * 2, self.kernel_size - 1,
                    padding="same", activation=tf.nn.relu)

            h = tfl.conv2d(
                    h, self.kernel_num, self.kernel_size - 1,
                    padding="same", activation=tf.nn.relu)
            h = tfl.max_pooling2d(h, 2, 2)
            # h = tfl.conv2d(
            #         h, self.kernel_num, self.kernel_size - 2,
            #         padding="same", activation=tf.nn.relu)

            h = tfl.flatten(h)
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            return z_mean

    def decode(self, z, x_dim):
        im_h, im_w, im_c = self.image_shape

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
            h = tfl.dense(
                    h, units=im_h // 4 * im_w // 4 * self.kernel_num,
                    activation=tf.nn.relu)
            h = tf.reshape(h, (-1, im_h // 4, im_w // 4, self.kernel_num))

            # h = tfl.conv2d_transpose(
            #         h, self.kernel_num, self.kernel_size - 2,
            #         strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, self.kernel_size - 1,
                    strides=2, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 2, self.kernel_size - 1,
                    strides=1, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, self.kernel_num * 4, self.kernel_size,
                    strides=2, padding="same", activation=tf.nn.relu)
            h = tfl.conv2d_transpose(
                    h, im_c, self.kernel_size,
                    strides=1, padding="same", activation=None)

            h = tfl.flatten(h)
            y_mean = h
            return y_mean


class FCClassifierCoder():
    def __init__(self, dataset, h_dim=400, layers_num=3):
        self.h_dim = h_dim
        self.image_shape = dataset.image_shape
        self.hidden_dims = [h_dim] * layers_num
        print("hidden dims", self.hidden_dims)

    def encode(self, x, y, z_dim, training=False):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            hd = self.hidden_dims[0]
            # TODO: to vs dodwanie
            x = tf.concat([x, y], 1)
            h = tf.nn.relu(tfl.dense(x, units=hd))
            for hd in self.hidden_dims[1:]:
                h = tfl.dense(h, units=hd, activation=tf.nn.relu)
                # h = tfl.batch_normalization(h, training=training)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            self.encoder_output = z_mean
            return z_mean

    def decode(self, z, y, x_dim, training=False):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = tf.concat([z, y], 1)
            for hd in reversed(self.hidden_dims):
                h = tfl.dense(h, units=hd, activation=tf.nn.relu)
            y_mean = tfl.dense(h, units=x_dim, name='y_mean', activation=tf.nn.sigmoid)
            self.decoder_output = y_mean
            return y_mean

class FCCoder():
    def __init__(self, dataset, h_dim=300, layers_num=2):
        self.h_dim = h_dim
        self.image_shape = dataset.image_shape
        self.hidden_dims = [h_dim] * layers_num
        
        if dataset.name == "svhn" and dataset.whitened:
            self.output_activation_fn = None
        else:
            self.output_activation_fn = tf.nn.sigmoid
        # self.hidden_dims[-1] = self.hidden_dims[-1] // 2

    def encode(self, x, z_dim, training=False):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            for hd in self.hidden_dims:
                h = tfl.dense(h, units=hd, activation=tf.nn.relu)
                # h = tfl.batch_normalization(h, training=training)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            return z_mean

    def decode(self, z, x_dim, training=False):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            for hd in reversed(self.hidden_dims):
                h = tfl.dense(h, units=hd, activation=tf.nn.relu)
                # h = tfl.batch_normalization(h, training=training)
            y_mean = tfl.dense(h, units=x_dim, name='y_mean', activation=self.output_activation_fn)
            return y_mean

class VaeFCCoder():
    def __init__(self, dataset, h_dim=300, layers_num=2):
        self.h_dim = h_dim
        self.image_shape = dataset.image_shape
        self.hidden_dims = [h_dim] * layers_num
        self.output_activation_fn = None if dataset.name == "svhn" else None
        # self.hidden_dims[-1] = self.hidden_dims[-1] // 2

    def encode(self, x, z_dim, training=False):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h = x
            for hd in self.hidden_dims:
                h = tfl.dense(h, units=hd, activation=tf.nn.relu)
                # h = tfl.batch_normalization(h, training=training)
            z_mean = tfl.dense(h, units=z_dim, name='z_mean')
            z_var = tfl.dense(h, units=z_dim, name='z_mean', activation=tf.nn.softplus)
            distr = MultivariateNormalDiag(loc=z_mean, scale_diag=z_var)
            # z_mean = tfl.batch_normalization(z_mean, training=training)
            return distr, distr.sample()

    def decode(self, z, x_dim, training=False):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            h = z
            for hd in reversed(self.hidden_dims):
                h = tfl.dense(h, units=hd, activation=tf.nn.relu)
                # h = tfl.batch_normalization(h, training=training)
            y_mean = tfl.dense(h, units=x_dim, name='y_mean', activation=self.output_activation_fn)
            return y_mean

class DummyClassifier():
    def __init__(self, *args, **kwargs):
        raise NotImplemented

    def encode(self, *args, **kwargs):
        raise NotImplemented

    def decode(self, *args, **kwargs):
        raise NotImplemented
