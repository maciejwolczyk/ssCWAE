from torch import nn

# VARIOUS ARCHITECTURES
# class CelebaCoder():
#     def __init__(
#             self, dataset, h_dim=256, kernel_size=4, kernel_num=32):
# 
#         self.h_dim = h_dim
#         self.kernel_size = kernel_size
#         self.kernel_num = kernel_num
#         self.image_shape = dataset.image_shape
# 
#     def encode(self, x, z_dim, training=False):
#         im_h, im_w, im_c = self.image_shape
# 
#         with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
#             h = x
#             h = tf.reshape(h, (-1, im_h, im_w, im_c))
# 
#             h = tfl.conv2d(
#                     h, self.kernel_num, self.kernel_size,
#                     strides=2, padding="same")
#             h = tf.nn.relu(h)
# 
#             h = tfl.conv2d(
#                     h, self.kernel_num, self.kernel_size,
#                     strides=2, padding="same")
#             h = tf.nn.relu(h)
# 
#             h = tfl.conv2d(
#                     h, self.kernel_num * 2, self.kernel_size,
#                     strides=2, padding="same")
#             h = tf.nn.relu(h)
# 
#             h = tfl.conv2d(
#                     h, self.kernel_num * 2, self.kernel_size,
#                     strides=2, padding="same")
#             h = tf.nn.relu(h)
# 
#             h = tfl.flatten(h)
#             h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
#             h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
#             z_mean = tfl.dense(h, units=z_dim, name='z_mean')
#             # z_mean = tfl.batch_normalization(z_mean, training=training)
#             return z_mean
# 
#     def decode(self, z, x_dim, training=False):
#         im_h, im_w, im_c = self.image_shape
# 
#         with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
#             h = z
#             h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
#             h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
#             stride = 16
#             h = tfl.dense(
#                     h, units=im_h // stride * im_w // stride * self.kernel_num * 2,
#                     activation=tf.nn.relu)
#             new_shape = (-1, im_h // stride, im_w // stride, self.kernel_num * 2)
#             h = tf.reshape(h, new_shape)
# 
#             h = tfl.conv2d_transpose(
#                     h, self.kernel_num * 2, self.kernel_size,
#                     strides=2, padding="same", activation=tf.nn.relu)
# 
#             h = tfl.conv2d_transpose(
#                     h, self.kernel_num, self.kernel_size,
#                     strides=2, padding="same", activation=tf.nn.relu)
# 
#             h = tfl.conv2d_transpose(
#                     h, self.kernel_num, self.kernel_size,
#                     strides=2, padding="same", activation=tf.nn.relu)
# 
#             h = tfl.conv2d_transpose(
#                     h, im_c, self.kernel_size,
#                     strides=2, padding="same", activation=tf.nn.sigmoid)
# 
#             h = tfl.flatten(h)
#             y_mean = h
#             return y_mean
# 
# 
# class CifarCoder():
#     def __init__(
#             self, dataset, h_dim=256, kernel_size=3, kernel_num=32):
# 
#         self.h_dim = h_dim
#         self.kernel_size = kernel_size
#         self.kernel_num = kernel_num
#         self.image_shape = dataset.image_shape
# 
#     def encode(self, x, z_dim, training=False):
#         im_h, im_w, im_c = self.image_shape
# 
#         with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
#             h = x
#             h = tf.reshape(h, (-1, im_h, im_w, im_c))
#             # TODO:
#             h = tfl.conv2d(
#                     h, self.kernel_num, self.kernel_size,
#                     strides=1, padding="same")
#             h = tf.nn.relu(h)
# 
#             h = tfl.conv2d(
#                     h, self.kernel_num, self.kernel_size,
#                     strides=2, padding="same")
#             h = tf.nn.relu(h)
# 
#             # TODO: tu troche zmienilem
#             h = tfl.conv2d(
#                     h, self.kernel_num * 2, self.kernel_size,
#                     strides=2, padding="same")
#             h = tf.nn.relu(h)
# 
#             h = tfl.conv2d(
#                     h, self.kernel_num * 2, self.kernel_size,
#                     strides=2, padding="same")
#             h = tf.nn.relu(h)
# 
#             h = tfl.flatten(h)
#             h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
#             z_mean = tfl.dense(h, units=z_dim, name='z_mean')
#             # z_mean = tfl.batch_normalization(z_mean, training=training)
#             return z_mean
# 
#     def decode(self, z, x_dim, training=False):
#         im_h, im_w, im_c = self.image_shape
# 
#         with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
#             h = z
#             h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
#             # h = tfl.dense(h, units=self.h_dim, activation=tf.nn.relu)
#             stride = 8
#             h = tfl.dense(
#                     h, units=im_h // stride * im_w // stride * self.kernel_num * 2,
#                     activation=tf.nn.relu)
#             new_shape = (-1, im_h // stride, im_w // stride, self.kernel_num * 2)
#             h = tf.reshape(h, new_shape)
# 
#             h = tfl.conv2d_transpose(
#                     h, self.kernel_num * 2, self.kernel_size,
#                     strides=2, padding="same", activation=tf.nn.relu)
# 
#             h = tfl.conv2d_transpose(
#                     h, self.kernel_num, self.kernel_size,
#                     strides=2, padding="same", activation=tf.nn.relu)
# 
#             h = tfl.conv2d_transpose(
#                     h, self.kernel_num, self.kernel_size,
#                     strides=2, padding="same", activation=tf.nn.relu)
# 
#             h = tfl.conv2d_transpose(
#                     h, im_c, self.kernel_size,
#                     strides=1, padding="same", activation=tf.nn.sigmoid)
# 
#             h = tfl.flatten(h)
#             y_mean = h
#             return y_mean


class FCCoder(nn.Module):
    def __init__(self, layers_num, input_dim, hidden_dim, latent_dim):
        super(FCCoder, self).__init__()

        encoder_modules = [nn.Linear(input_dim, hidden_dim)]
        for layer_idx in range(layers_num - 2):
            encoder_modules += [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
        encoder_modules += [nn.ReLU(), nn.Linear(hidden_dim, latent_dim)]
        self.encoder = nn.Sequential(*encoder_modules)

        decoder_modules = [nn.Linear(latent_dim, hidden_dim)]
        for layer_idx in range(layers_num - 2):
            decoder_modules += [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
        decoder_modules += [nn.ReLU(), nn.Linear(hidden_dim, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
