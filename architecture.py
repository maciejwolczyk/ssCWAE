import tensorflow as tf
import tensorflow.layers as tfl


# VARIOUS ARCHITECTURES
class CelebaCoder():
    def __init__(
            self, dataset, h_dim=256, kernel_size=4, kernel_num=32):

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
            new_shape = (-1, im_h // stride, im_w // stride, self.kernel_num * 2)
            h = tf.reshape(h, new_shape)

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
    def __init__(
            self, dataset, h_dim=256, kernel_size=3, kernel_num=32):

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
            new_shape = (-1, im_h // stride, im_w // stride, self.kernel_num * 2)
            h = tf.reshape(h, new_shape)

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
            y_mean = tfl.dense(
                    h, units=x_dim,
                    name='y_mean', activation=self.output_activation_fn)
            return y_mean
