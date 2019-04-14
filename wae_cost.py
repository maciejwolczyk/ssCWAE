import numpy as np
import tensorflow as tf

epsilon_value = 0.00001 

def norm_of_diff(x: tf.Tensor, y: tf.Tensor):
    epsilon = tf.constant(epsilon_value)
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(x, 0), tf.expand_dims(y, 1))), axis = 2)+epsilon)

def mmd_penalty(sample_qz, sample_pz, kernel: str='IMQ'):
    epsilon = tf.constant(epsilon_value)
    
    n = tf.cast(tf.shape(sample_qz)[0], tf.float32)
    d = tf.cast(tf.shape(sample_qz)[1], tf.float32)
    
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)
    
    if kernel == 'IMQ':
        sigma2_p = 1. ** 2
        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keepdims=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * tf.matmul(sample_pz, sample_pz, transpose_b=True)

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * tf.matmul(sample_qz, sample_qz, transpose_b=True)

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods
        
        # k(x, y) = C / (C + ||x - y||^2)
        # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        #if opts['pz'] == 'normal':
        Cbase = 2. * d * sigma2_p
        #elif opts['pz'] == 'sphere':
        #    Cbase = 2.
        #elif opts['pz'] == 'uniform':
            # E ||x - y||^2 = E[sum (xi - yi)^2]
            #               = zdim E[(xi - yi)^2]
            #               = const * zdim
        #    Cbase = opts['zdim']
        stat = 0.
        TempSubtract = 1. - tf.eye(n)
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz) + C / (C + distances_pz)
            res1 = tf.multiply(res1, TempSubtract)
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
        return stat
    else:
        kernel_function = None
        print(kernel)
        if kernel == 'jacek_regular':
            print('Using JACEK REGULAR')
            kernel_function = lambda x, y: -1.0 * norm_of_diff(x, y)
        elif kernel == 'jacek_sqrt':
            kernel_function = lambda x, y: -1.0 * tf.sqrt(norm_of_diff(x, y) + epsilon)

        NoDiagonal = 1. - tf.eye(n)
        res1 = kernel_function(sample_qz, sample_qz) + kernel_function(sample_pz, sample_pz)
        res1 = tf.multiply(res1, NoDiagonal)
        res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        res2 = kernel_function(sample_qz, sample_pz)
        res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        return res1 - res2
    

class WAECost():

    def __init__(self, kernel_name: str = 'IMQ'):
        self.__kernel_name = kernel_name

    def evaluate(self, tensor_z, z_dim):
        dist = tf.distributions.Normal(np.zeros(z_dim, dtype=np.float32), np.ones(z_dim, dtype=np.float32))
        tensor_input_latent_sample = dist.sample(tf.shape(tensor_z)[0])          
        return mmd_penalty(tensor_z, tensor_input_latent_sample, self.__kernel_name)
    