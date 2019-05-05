import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp
import tensorflow.layers as tfl
import tensorflow.train as tft
from math import pi

import architecture
from wae_cost import WAECost


def positive_kernel(X, Y):
    X = tf.expand_dims(X, 0)
    Y = tf.expand_dims(Y, 1)
    return -1.0 * tf.norm(X - Y, axis=2, ord=2)

def conditionally_positive_distance(X):
    normal_sample = tf.random.normal(shape=tf.shape(X))

    kernel_function = positive_kernel
    # kernel_function = lambda x, y: -1.0 * tf.sqrt(tf.abs(x - y))

    n = tf.shape(X)[0]
    no_diagonal_mask = tf.cast(1 - tf.eye(n), tf.bool)
    A = tf.boolean_mask(kernel_function(X, X), no_diagonal_mask)
    A = tf.reduce_mean(A)
    B = kernel_function(normal_sample, normal_sample)
    B = tf.reduce_mean(tf.boolean_mask(B, no_diagonal_mask))

    C = kernel_function(X, normal_sample)
    C = 2. * tf.reduce_mean(C)
    return A + B - C

def gmm_conditionally_positive_distance(X, means, z_dim):
    normal_sample = tf.random.normal(shape=tf.shape(X))
    n = tf.shape(X)[0]

    examples_per_class = n // 10
    tf.shape(X)

    # means: [classes_num, z_dim]
    means = tf.expand_dims(means, 0)
    means = tf.tile(means, (examples_per_class, 1, 1))
    means = tf.reshape(means, (-1, z_dim))

    normal_sample = normal_sample + means

    kernel_function = positive_kernel
    # kernel_function = lambda x, y: -1.0 * tf.sqrt(tf.abs(x - y))

    no_diagonal_mask = tf.cast(1 - tf.eye(n), tf.bool)
    A = tf.boolean_mask(kernel_function(X, X), no_diagonal_mask)
    A = tf.reduce_mean(A)
    B = kernel_function(normal_sample, normal_sample)
    B = tf.reduce_mean(tf.boolean_mask(B, no_diagonal_mask))

    C = kernel_function(X, normal_sample)
    C = 2. * tf.reduce_mean(C)
    return A + B - C


def cramer_wold_distance(X, m, alpha, p, gamma, weighted=False):
    N = tf.cast(tf.shape(X)[0], tf.float32)
    D = tf.cast(tf.shape(X)[1], tf.float32)
    # N0 = 73257
    N0 = N

    alpha_matrix = tf.expand_dims(alpha, 0) + tf.expand_dims(alpha, 1)

    X_sub_matrix = tf.subtract(tf.expand_dims(X, 0), tf.expand_dims(X, 1))
    A1 = norm_squared(X_sub_matrix, axis=2)
    # r = tf.reduce_sum(X * X, -1)
    # r = tf.reshape(r, [-1, 1])
    # A1 = r - 2 * tf.matmul(X, tf.transpose(X)) + tf.transpose(r)

    if weighted:
        N = 1

    A1 = tf.reduce_sum(phi_d(A1 / (4 * gamma), D))
    A = 1/(N*N * tf.sqrt(2 * pi * 2 * gamma)) * A1

    m_sub_matrix = tf.subtract(tf.expand_dims(m, 0), tf.expand_dims(m, 1))
    p_mul_matrix = tf.matmul(tf.expand_dims(p, 1), tf.expand_dims(p, 0))
    B1 = norm_squared(m_sub_matrix, axis=2)
    B2 = phi_d(B1 / (2 * alpha_matrix + 4 * gamma), D)
    B3 = p_mul_matrix / tf.sqrt(2 * pi * (alpha_matrix + 2 * gamma))
    B = tf.reduce_sum(B3 * B2)

    m_X_sub_matrix = tf.subtract(tf.expand_dims(m, 0), tf.expand_dims(X, 1))
    C1 = norm_squared(m_X_sub_matrix, axis=2)
    C2 = phi_d(C1 / (2 * (alpha + 2 * gamma)), D)
    class_logits = C2
    C3 = 2 * p / (N * tf.sqrt(2 * pi * (alpha + 2 * gamma))) * C2
    C = tf.reduce_sum(C3)
    return tf.reduce_mean(A + B - C)

def single_cramer_wold_distance(X, gamma):
    N = tf.cast(tf.shape(X)[0], tf.float32)
    D = tf.cast(tf.shape(X)[1], tf.float32)
    # N0 = 73257
    N0 = N

    alpha = 1

    X_sub_matrix = tf.subtract(tf.expand_dims(X, 0), tf.expand_dims(X, 1))
    A1 = norm_squared(X_sub_matrix, axis=2)

    A1 = tf.reduce_sum(phi_d(A1 / (4 * gamma), D))
    A = 1/(N*N * tf.sqrt(2 * pi * 2 * gamma)) * A1

    B3 = 1 / tf.sqrt(2 * pi * (alpha + 2 * gamma))
    B = tf.reduce_sum(B3)

    C1 = norm_squared(X, axis=1)
    C2 = phi_d(C1 / (2 * (alpha + 2 * gamma)), D)
    class_logits = C2
    C3 = 2 / (N * tf.sqrt(2 * pi * (alpha + 2 * gamma))) * C2
    C = tf.reduce_sum(C3)
    return tf.reduce_mean(A + B - C)


class DeepGmmModel():
    def __init__(
            self, name, coder, dataset, m1=None,
            z_dim=300, supervised_weight=1.0,
            learning_rate=1e-3, cw_weight=1.0,
            init=1.0, gamma_weight=1.0):

        tf.reset_default_graph()
        self.name = name
        self.init = init
        self.gamma_weight = gamma_weight
        self.coder = coder
        self.optimizer = tft.AdamOptimizer(learning_rate)
        self.m1 = m1

        self.z_dim = z_dim

        if m1 is not None:
            x_dim = m1.z_dim
        else:
            x_dim = dataset.x_dim


        tensor_x = tf.placeholder(
                shape=[None, x_dim],
                dtype=tf.float32, name='input_x')
        tensor_labels = tf.placeholder(
                shape=[None, dataset.classes_num],
                dtype=tf.float32, name='target_y')
        tensor_training = tf.placeholder_with_default(False, shape=[])
        train_labeled = tf.placeholder_with_default(True, shape=[])

        labeled_mask = get_labels_mask(tensor_labels)
        classifier = EntropyClassifier(
            tensor_x, tensor_labels, labeled_mask,
            dataset, tensor_training)

        classes_probs = classifier.out["probs"]
        means, variances, probs = get_gaussians(
            z_dim, init, dataset, dataset.classes_num)

        N0 = tf.shape(tensor_x)[0]
        gamma = tf.cast(tf.pow(4 / (3 * N0 / dataset.classes_num), 0.4), tf.float32)
        gamma *= gamma_weight


        rec_costs = []
        cw_costs = []
        # for class_idx in range(dataset.classes_num):
        #     one_hot = np.zeros((1, dataset.classes_num))
        #     one_hot[0, class_idx] = 1
        #     one_hot = tf.constant(one_hot, dtype=tf.float32)

        #     class_probs = tf.transpose(classes_probs)[class_idx]

        #     y_hot = tf.tile(one_hot, (N0, 1))
        #     tensor_z = coder.encode(tensor_x, y_hot, z_dim, tensor_training)
        #     tensor_y = coder.decode(tensor_z, y_hot, x_dim, tensor_training)

        #     normalized_class_probs = class_probs / tf.reduce_sum(class_probs)
        #     cw_tensor_z = tensor_z * tf.reshape(normalized_class_probs, (-1, 1))

        #     cw_cost = cramer_wold_distance(
        #         cw_tensor_z, means, variances, probs, gamma, weighted=True)
        #     cw_cost = tf.log(cw_cost)
        #     cw_costs += [cw_cost]

        #     if dataset.name == "mnist":
        #         rec_cost = tensor_x * tf.log(tensor_y + 1e-8) + (1 - tensor_x) * tf.log(1 - tensor_y + 1e-8)
        #         rec_cost = -tf.reduce_sum(rec_cost, axis=-1) * class_probs
        #     else:
        #         rec_cost = norm_squared(tensor_x - tensor_y, axis=-1) * class_probs

        #     rec_costs += [rec_cost]
        # full_cw_cost = tf.stack(cw_costs) # [10]
        # full_cw_cost = tf.reduce_mean(full_cw_cost) # [1]

        # full_rec_cost = tf.stack(rec_costs) # [10, N]
        # full_rec_cost = tf.reduce_sum(full_rec_cost, 0) # [10]
        # full_rec_cost = tf.reduce_mean(full_rec_cost)

        tensor_z = coder.encode(tensor_x, classes_probs, z_dim, tensor_training)
        tensor_y = coder.decode(tensor_z, classes_probs, x_dim, tensor_training)

        full_cw_cost = cramer_wold_distance(
            tensor_z, means, variances, probs, gamma, weighted=False)
        full_cw_cost = tf.log(full_cw_cost)

        if dataset.name == "mnist":
            rec_cost = tensor_x * tf.log(tensor_y + 1e-8) + (1 - tensor_x) * tf.log(1 - tensor_y + 1e-8)
            rec_cost = -tf.reduce_sum(rec_cost, axis=-1)
        else:
            rec_cost = norm_squared(tensor_x - tensor_y, axis=-1)
        full_rec_cost = tf.reduce_mean(rec_cost)


        gmm_class_logits = calculate_logits(
            tensor_z, means, variances, probs) 
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=gmm_class_logits, labels=classes_probs)

        distance_cost = linear_distance_penalty(
                z_dim, means, variances, probs, dataset.classes_num)

        full_cost = (full_rec_cost
                     + full_cw_cost
                     + cross_entropy
                     - classifier.costs["entropy"]
                     + classifier.costs["class"] * supervised_weight)

        full_train_op = self.optimizer.minimize(full_cost)


        samples_z = self.coder.encode(tensor_x, tensor_labels, z_dim, False)
        samples_y = self.coder.decode(samples_z, tensor_labels, x_dim, False)

        self.saver = tf.train.Saver(max_to_keep=10000)
        self.preds = classes_probs
        self.placeholders = {
            "X": tensor_x,
            "y": tensor_labels,
            "train_labeled": train_labeled,
            "training": tensor_training,
        }

        self.costs = {
            "full": full_cost,
            "class": classifier.costs["class"],
            "reconstruction": full_rec_cost,
            "distance": distance_cost,
            "cw": full_cw_cost
        }

        self.train_ops = {
            "full": full_train_op
        }

        self.out = {
            "z": samples_z,
            "y": samples_y
        }

        self.gausses = {
            "means": means,
            "variations": variances,
            "probs": probs
        }

class EntropyClassifier:
    def __init__(
            self, tensor_x, tensor_labels,
            labeled_mask, dataset, tensor_training,
            labeled_super_weight=1.0):

        with tf.variable_scope("entropy_classifier"):
            self.labeled_super_weight = labeled_super_weight
            self.dataset = dataset
            self.tensor_training = tensor_training
            self.build_graph(tensor_x, tensor_labels, labeled_mask)


    def build_graph(self, tensor_x, tensor_labels, labeled_mask):
        tensor_logits = self.build_net(tensor_x)
        tensor_probs = tf.nn.softmax(tensor_logits)

        labeled_logits = tf.boolean_mask(tensor_logits, labeled_mask)
        labeled_labels = tf.boolean_mask(tensor_labels, labeled_mask)

        class_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labeled_labels, logits=labeled_logits)
        class_cost = tf.reduce_mean(class_cost)

        # minimalizujemy czy maksymalizujemy?
        entropy_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tensor_probs, logits=tensor_logits)
        entropy_cost = tf.reduce_mean(class_cost)


        full_cost = self.labeled_super_weight * class_cost + entropy_cost  # weighted?

        tensor_logits = tf.cond(
                self.tensor_training,
                lambda: tf.where(labeled_mask, tensor_labels, tensor_logits),
                lambda: tensor_logits)
        tensor_probs = tf.cond(
                self.tensor_training,
                lambda: tf.where(labeled_mask, tensor_labels, tf.nn.softmax(tensor_logits)),
                lambda: tf.nn.softmax(tensor_logits))

        # tensor_probs = tf.nn.softmax(tensor_logits)

        self.costs = {
            "class": class_cost,
            "entropy": entropy_cost,
            "full": full_cost
        }

        self.out = {
            "logits": tensor_logits,
            "probs": tensor_probs
        }

        return full_cost

    def build_net(self, input_x):
        h_dim = 500
        x = tfl.dense(input_x, h_dim, activation=tf.nn.relu)
        x = tfl.dense(x, h_dim, activation=tf.nn.relu)
        # x = tfl.dense(x, h_dim, activation=tf.nn.relu)
        logits = tfl.dense(x, self.dataset.classes_num, activation=None)
        return logits


class CwaeClassifier:
    def __init__(self, tensor_x, tensor_labels, train_labeled, labeled_mask,
                 dataset, tensor_distance_weight, tensor_training,
                 z_dim=50):
        with tf.variable_scope("gmm_classifier"):
            h_dim = 300
            layers_num = 3
            self.dataset = dataset
            self.coder = architecture.FCCoder(
                dataset, h_dim=h_dim, layers_num=layers_num)
            self.build_graph(
                tensor_x, tensor_labels, train_labeled, labeled_mask,
                tensor_distance_weight, tensor_training, z_dim)

    def build_graph(
            self, tensor_x, tensor_labels, train_labeled, labeled_mask,
            tensor_distance_weight, tensor_training, z_dim):

        N0 = tf.shape(tensor_x)[0]
        # D = tf.shape(tensor_x)[-1]
        gamma = tf.cast(tf.pow(4 / (3 * N0 / self.dataset.classes_num), 0.4), tf.float32)
        means, variances, probs = get_gaussians(z_dim, 1.0, self.dataset)

        tensor_z = self.coder.encode(tensor_x, z_dim)
        tensor_y = self.coder.decode(tensor_z, self.dataset.x_dim)

        unsupervised_tensor_z = tf.cond(
            train_labeled,
            lambda: tensor_z,
            lambda: tf.boolean_mask(tensor_z, tf.logical_not(labeled_mask))
        )

        rec_cost = norm_squared(tensor_x - tensor_y, axis=-1)
        cw_cost = cramer_wold_distance(
                unsupervised_tensor_z, means, variances, probs, gamma)
        log_cw_cost = tf.log(cw_cost)

        tensor_logits = calculate_logits(tensor_z, means, variances, probs)
        tensor_probs = tf.nn.softmax(tensor_logits)

        class_cost = calculate_logits_cost(
            tensor_logits, tensor_labels, labeled_mask)
        dist_cost = linear_distance_penalty(
            z_dim, means, variances, probs, self.dataset.classes_num)

        print("CLASSIFIER")
        rec_cost = tf.cond(
            train_labeled,
            lambda: tf.reduce_mean(rec_cost),
            lambda: tf.reduce_mean(tf.boolean_mask(rec_cost, tf.logical_not(labeled_mask)))
        )
        full_cost = rec_cost + log_cw_cost + class_cost + tensor_distance_weight * dist_cost
        print(full_cost)

        tensor_logits = tf.cond(
                tensor_training,
                lambda: tf.where(labeled_mask, tensor_labels, tensor_logits),
                lambda: tensor_logits)
        tensor_probs = tf.cond(
                tensor_training,
                lambda: tf.where(labeled_mask, tensor_labels, tf.nn.softmax(tensor_logits)),
                lambda: tf.nn.softmax(tensor_logits))

        self.costs = {
            "full": full_cost,
            "class": class_cost,
            "cw_cost": cw_cost,
            "reconstruction": rec_cost,
            "distance": dist_cost
        }

        self.out = {
            "logits": tensor_logits,
            "probs": tensor_probs
        }

        return full_cost


class CwaeModel():

    def __init__(
        self, name, coder, dataset, latent_dim=300,
        learning_rate=1e-3, gamma_weight=1.0, cw_weight=1.0):


        self.name = name
        self.z_dim = latent_dim
        tf.reset_default_graph()

        optimizer = tft.AdamOptimizer(learning_rate)
        tensor_x = tf.placeholder(
                shape=[None, dataset.x_dim],
                dtype=tf.float32, name='input_x')
        tensor_labels = tf.placeholder(
                shape=[None, dataset.classes_num],
                dtype=tf.float32, name='target_y')
        train_labeled = tf.placeholder_with_default(True, shape=[])
        tensor_training = tf.placeholder_with_default(False, shape=[])

        labeled_mask = get_labels_mask(tensor_labels)
        means, variances, probs = get_gaussians(latent_dim, 0., dataset, 1)

        tensor_z = coder.encode(tensor_x, latent_dim, tensor_training)
        tensor_y = coder.decode(tensor_z, dataset.x_dim, tensor_training)

        rec_cost = norm_squared(tensor_x - tensor_y, axis=-1)
        rec_cost = tf.cond(
            train_labeled,
            lambda: tf.reduce_mean(rec_cost),
            lambda: tf.reduce_mean(
                tf.boolean_mask(rec_cost, tf.logical_not(labeled_mask)))
        )

        unsupervised_tensor_z = tf.cond(
            train_labeled,
            lambda: tensor_z,
            lambda: tf.boolean_mask(tensor_z, tf.logical_not(labeled_mask))
        )

        N0 = tf.shape(unsupervised_tensor_z)[0]
        gamma = tf.cast(tf.pow(4 / (3 * N0), 0.4), tf.float32)
        gamma *= gamma_weight

        # TODO: log_cw
        # cw_cost = single_cramer_wold_distance(unsupervised_tensor_z, gamma)
        cw_cost = WAECost("jacek_regular").evaluate(unsupervised_tensor_z, latent_dim)
        log_cw_cost = cw_weight * tf.log(cw_cost)

        cpd_cost = conditionally_positive_distance(unsupervised_tensor_z)
        cpd_cost *= cw_weight

        full_cost = tf.reduce_mean(rec_cost + cw_cost)
        train_op = optimizer.minimize(full_cost)

        full_cpd_cost = tf.reduce_mean(rec_cost + cpd_cost)
        cpd_train_op = optimizer.minimize(full_cpd_cost)

        self.saver = tf.train.Saver(max_to_keep=10000)
        self.placeholders = {
            "X": tensor_x,
            "y": tensor_labels,
            "train_labeled": train_labeled,
            "training": tensor_training,
        }

        self.costs = {
            "full": full_cost,
            "full_cpd": full_cpd_cost,
            "reconstruction": rec_cost,
            "cw": log_cw_cost,
            "cpd": cpd_cost
        }

        self.train_ops = {
            "full": train_op,
            "full_cpd": cpd_train_op
        }

        self.gausses = {
            "means": means,
            "variations": variances,
            "probs": probs
        }
        self.out = {
            "z": tensor_z,
            "y": tensor_y
        }


class GmmCwaeModel():
    def __init__(
        self, name, coder, dataset, m1=None,
        z_dim=300, supervised_weight=1.0, distance_weight=1.0,
        erf_weight=1.0, erf_alpha=0.05,
        learning_rate=1e-3, cw_weight=1.0,
        classifier_cls=EntropyClassifier, classifier_distance_weight=1.0,
        eps=1e-2, init=1.0, gamma_weight=1.0, labeled_super_weight=2.0):

        tf.reset_default_graph()
        self.name = name
        self.init = init
        self.gamma_weight = gamma_weight
        self.coder = coder
        optimizer = tft.AdamOptimizer(learning_rate)
        self.optimizer = optimizer
        self.classifier_cls = classifier_cls
        self.m1 = m1

        self.cw_weight = cw_weight

        self.z_dim = z_dim

        if m1 is not None:
            x_dim = m1.z_dim
        else:
            x_dim = dataset.x_dim


        # self.prepare_placeholders()
        # Prepare placeholders
        tensor_x = tf.placeholder(
                shape=[None, x_dim],
                dtype=tf.float32, name='input_x')
        tensor_labels = tf.placeholder(
                shape=[None, dataset.classes_num],
                dtype=tf.float32, name='target_y')

        train_labeled = tf.placeholder_with_default(True, shape=[])
        tensor_distance_weight = tf.placeholder_with_default(distance_weight, shape=[])
        tensor_classifier_distance_weight = tf.placeholder_with_default(classifier_distance_weight, shape=[])
        tensor_supervised_weight = tf.placeholder_with_default(supervised_weight, shape=[])
        tensor_cw_weight = tf.placeholder_with_default(cw_weight, shape=[])
        tensor_erf_weight = tf.placeholder_with_default(erf_weight, shape=[])
        tensor_training = tf.placeholder_with_default(False, shape=[])

        labeled_mask = get_labels_mask(tensor_labels)


        if classifier_cls != architecture.DummyClassifier:
            classifier = self.classifier_cls(
                    tensor_x, tensor_labels, labeled_mask,
                    dataset, tensor_training,
                    labeled_super_weight=labeled_supervised_weight)
                    # tensor_classifier_distance_weight)
            class_cost = classifier.costs["full"]
            class_logits = classifier.out["logits"]
            class_probs = classifier.out["probs"]


            # TODO: brudny hack
            y_dist = tf.distributions.Categorical(probs=[0.1] * 10)
            simplex_y = y_dist.sample(1)
            simplex_y = tf.cast(tf.one_hot(simplex_y, dataset.classes_num), tf.float32)
            simplex_y = tf.reshape(simplex_y, (-1,))

            # y_dist = tf.distributions.Dirichlet([1.0] * 10)
            # simplex_y = y_dist.sample(1)
            # simplex_y = tf.reshape(simplex_y, (-1,))

            sigma = 0.2 # tf.constant([0.1])
            norm_dist = tf.contrib.distributions.MultivariateNormalDiag(
                simplex_y, [sigma] * dataset.classes_num)
            norm_probs = norm_dist.prob(class_probs) # [batch_size, class_num]
            norm_probs = tf.reshape(norm_probs, (-1,))
            tensor_norm_probs = tf.reshape(norm_probs, (-1, 1))
            # norm_weights = norm_probs / tf.reduce_sum(norm_probs)

            # TODO: kontrolne
            # simplex_y = tf.constant([0.1] * 10)

            classifier_probs = tensor_norm_probs
            # classifier_probs = class_probs

        else:
            examples_num = tf.shape(tensor_x)[0]
            simplex_y = tf.constant([0.1] * dataset.classes_num)
            norm_probs = tf.ones((examples_num,), dtype=tf.float32) / tf.cast(examples_num, tf.float32)


        # Coder and decoder
        # TODO: bardzo glupie, N0 nie moze byc liczone na podstawie tensor_x,
        # bo nie mamy go przy samplowaniu
        N0 = tf.cast(tf.shape(tensor_x)[0], tf.float32)
        tensor_simplex_y = tf.tile(tf.reshape(simplex_y, (1, -1)), (N0, 1))
        tensor_z = coder.encode(tensor_x, z_dim, tensor_training)

        N0 = tf.cast(tf.shape(tensor_z)[0], tf.float32)
        tensor_simplex_y = tf.tile(tf.reshape(simplex_y, (1, -1)), (N0, 1))
        tensor_y = coder.decode(tensor_z, x_dim, tensor_training)

        # Unsupervised examples are treated differently than supervised:
        unsupervised_tensor_z = tf.cond(
            train_labeled,
            lambda: tensor_z,
            lambda: tf.boolean_mask(tensor_z, tf.logical_not(labeled_mask)))
        N0 = tf.shape(unsupervised_tensor_z)[0]

        means, variances, probs = get_gaussians(z_dim, init, dataset, dataset.classes_num)

        gamma = tf.cast(tf.pow(4 / (3 * N0 / dataset.classes_num), 0.4), tf.float32)
        gamma *= self.gamma_weight

        class_logits = calculate_logits(
            tensor_z, means, variances, probs)
        class_probs = tf.nn.softmax(class_logits)
        class_cost = calculate_logits_cost(class_logits,
                tensor_labels, labeled_mask)
        # class_cost = self.cwae_class_cost(
        #     tensor_z, tensor_labels, labeled_mask,
        #     means, variances, probs, dataset, gamma)



        # Old Cramer-Wold cost
        cw_cost = cramer_wold_distance(
                unsupervised_tensor_z, means, variances, probs, gamma)
        # cw_cost = tf.Print(cw_cost, [cw_cost])
        log_cw_cost = tf.log(cw_cost)
        log_cw_cost *= tensor_cw_weight

        # TODO:
        cpd_cost = gmm_conditionally_positive_distance(unsupervised_tensor_z, means, z_dim)
        # log_cw_cost = cpd_cost

        rec_cost = norm_squared(tensor_x - tensor_y, axis=-1)
        log_rec_cost = tf.cond(
            train_labeled,
            lambda: tf.reduce_mean(tf.log(rec_cost)),
            lambda: tf.reduce_mean(
                tf.boolean_mask(tf.log(rec_cost), tf.logical_not(labeled_mask))
            )
        )

        rec_cost = tf.cond(
            train_labeled,
            lambda: tf.reduce_mean(rec_cost),
            lambda: tf.reduce_mean(
                tf.boolean_mask(rec_cost, tf.logical_not(labeled_mask)))
        )
        # rec_cost /= 100

        cec_cost = self.ceclike_class_cost(
            tensor_z, tensor_labels, labeled_mask,
            class_logits, means, variances, probs,
            dataset, training_mode=tensor_training)

        gmm_cost = self.gmm_imitation_cost(
            tensor_z, class_probs, means, variances, probs,
            dataset, training_mode=tensor_training)
        gmm_weight = 10.
        gmm_cost = gmm_weight * gmm_cost

        norm_cost = self.norm_imitation_cost(
            tensor_z, norm_probs,
            dataset, training_mode=tensor_training)
        norm_weight = 10.
        norm_cost = norm_weight * norm_cost
        erf_cost = self.total_erf(
            means, probs,
            alpha=erf_alpha, classes_num=dataset.classes_num
        )

        t_vars  = [t for t in tf.trainable_variables() if "bias" not in t.name]
        reg_cost = tf.add_n([tf.nn.l2_loss(v) for v in t_vars]) * 0.5 / dataset.train_examples_num


        distance_cost = linear_distance_penalty(
                z_dim, means, variances, probs, dataset.classes_num)
        entropy_cost = calculate_logits_entropy(class_logits, labeled_mask)


        # self.prepare_costs(
        #     rec_cost, log_cw_cost, distance_cost, cec_cost):

        unsupervised_cost = tf.reduce_mean(
                rec_cost
                + log_cw_cost
                + tensor_distance_weight * distance_cost)

        full_gmm_cost = tf.reduce_mean(
                rec_cost
                + gmm_cost
                # + log_cw_cost
                + tensor_supervised_weight * class_cost
                # + tensor_distance_weight * distance_cost
                + tensor_erf_weight * erf_cost)

        full_norm_cost = tf.reduce_mean(
                rec_cost
                + norm_cost
                + tensor_supervised_weight * class_cost
        )

        full_cost = tf.reduce_mean(
                rec_cost
                + log_cw_cost
                + tensor_supervised_weight * class_cost
                + tensor_distance_weight * distance_cost
                # + reg_cost
                # - entropy_cost
                )

        full_cec_cost = tf.reduce_mean(
                rec_cost
                + tensor_distance_weight * distance_cost
                + tensor_supervised_weight * cec_cost)

        full_cec_erf_cost = tf.reduce_mean(

                rec_cost
                + tensor_distance_weight * distance_cost
                + tensor_supervised_weight * cec_cost
                + tensor_erf_weight * erf_cost)

        tvars = tf.trainable_variables()
        freeze_vars = [
                var for var in tvars if
                "gmm_means" not in var.name
                and "gmm_betas" not in var.name]
        means_vars = [
                var for var in tvars if
                "gmm_means" in var.name
                or "gmm_betas" in var.name]

        class_vars = [
            var for var in tvars if
            "gmm_classifier" in var.name
        ]

        non_class_vars = [
            var for var in tvars if
            "gmm_classifier" not in var.name
        ]

        if classifier_cls != architecture.DummyClassifier:
            self.class_saver = tf.train.Saver(class_vars)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Prepare various train ops
            grads, gvars = zip(*optimizer.compute_gradients(full_cost))
            print(grads)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            print(grads)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in zip(grads, gvars)]
            train_op = optimizer.apply_gradients(capped_gvs)

            # train_op = optimizer.minimize(full_cost)
            freeze_train_op = optimizer.minimize(full_cost, var_list=freeze_vars)
            # means_train_op = optimizer.minimize(class_cost, var_list=means_vars)

            class_train_op = optimizer.minimize(class_cost)
            rec_train_op = optimizer.minimize(rec_cost)
            rec_dkl_train_op = optimizer.minimize(rec_cost + log_cw_cost)
            cw_train_op = optimizer.minimize(log_cw_cost)
            supervised_train_op = optimizer.minimize(class_cost)

            full_cec_train_op = optimizer.minimize(full_cec_cost)
            full_cec_erf_train_op = optimizer.minimize(full_cec_erf_cost)

            full_norm_train_op = optimizer.minimize(full_norm_cost)
            full_norm_freeze_train_op = optimizer.minimize(
                full_norm_cost, var_list=non_class_vars)
            full_gmm_train_op = optimizer.minimize(full_gmm_cost)
            full_gmm_freeze_train_op = optimizer.minimize(
                full_gmm_cost, var_list=non_class_vars)

        # Prepare variables for outside use
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.placeholders = {
            "X": tensor_x,
            "y": tensor_labels,
            "train_labeled": train_labeled,
            "distance_weight": tensor_distance_weight,
            "classifier_distance_weight": tensor_distance_weight,
            "erf_weight": tensor_erf_weight,
            "supervised_weight": tensor_supervised_weight,
            "cw_weight": tensor_cw_weight,
            "training": tensor_training,
        }

        self.out = {
            "logits": class_logits,
            "probs": class_probs,
            "z": tensor_z,
            "y": tensor_y,
            "simplex_y": simplex_y
        }

        self.gausses = {
            "means": means,
            "variations": variances,
            "probs": probs}

        self.costs = {
            "class": class_cost,
            "cw": log_cw_cost,
            "reconstruction": rec_cost,
            "distance": distance_cost,
            "cec": cec_cost,
            "gmm": gmm_cost,
            "norm": norm_cost,
            "erf": erf_cost,
            "cpd": cpd_cost,
            "full": full_cost,
            "unsupervised": unsupervised_cost,
            "entropy": -entropy_cost
        }

        self.train_ops = {
            "full": train_op,
            "full_freeze": freeze_train_op,
            "supervised": supervised_train_op,
            "rec": rec_train_op,
            "class": class_train_op,
            # "means_only": means_train_op,
            "cw": cw_train_op,
            "rec_dkl": rec_dkl_train_op,
            "full_cec": full_cec_train_op,
            "full_cec_erf": full_cec_erf_train_op,
            "full_gmm": full_gmm_train_op,
            "full_gmm_freeze": full_gmm_freeze_train_op,
            "full_norm": full_norm_train_op,
            "full_norm_freeze": full_norm_freeze_train_op
        }

        self.train_op = train_op
        self.freeze_train_op = freeze_train_op
        self.supervised_train_op = supervised_train_op
        self.preds = class_logits


    def pairwise_erf(self, first_mean, first_prob,
                     second_mean, second_prob, alpha):
        # transformation here
        n_dim = tf.cast(tf.shape(first_mean)[-1], tf.float32)
        v = (second_mean - first_mean) / tf.norm(first_mean - second_mean)
        first_mean = tf.tensordot(v, first_mean, 1)
        second_mean = tf.tensordot(v, second_mean, 1)

        mean_diff = second_mean - first_mean

        boundary = 1 / mean_diff * tf.log(first_prob / second_prob)
        boundary += (first_mean + second_mean) / 2

        first_erf = tf.erf((boundary - first_mean) / tf.sqrt(2.))
        second_erf = tf.erf((boundary - second_mean) / tf.sqrt(2.))
        total_error = (first_prob * (1 - 0.5 * (1 + first_erf))
                       + second_prob * 0.5 * (1 + second_erf))
        # total_error = tf.Print(total_error, [total_error])
        return tf.log(tf.reduce_max((alpha, total_error)))

    def total_erf(self, means, probs, classes_num=10, alpha=0.05):
        total = tf.zeros([])
        for first_idx in range(classes_num):
            first_mean = means[first_idx]
            first_prob = probs[first_idx]
            for second_idx in range(first_idx + 1, classes_num):
                second_mean = means[second_idx]
                second_prob = probs[second_idx]
                total += self.pairwise_erf(
                    first_mean, first_prob, second_mean, second_prob, alpha)
        return total / (classes_num * (classes_num - 1))

    def norm_imitation_cost(
            self, tensor_z, norm_probs,
            dataset, training_mode):

        N0 = tf.cast(tf.shape(tensor_z)[0], tf.float32)
        D = tf.cast(tf.shape(tensor_z)[-1], tf.float32)
        gamma = tf.pow(4 / (3 * N0), 0.4) * self.gamma_weight
        # TODO: sanity check
        # norm_probs = tf.ones_like(norm_probs)
        norm_weights = norm_probs / tf.reduce_sum(norm_probs)

        # Wzor na odleglosc N(0,1) do w_i
        z_matrix = tf.expand_dims(tensor_z, 1) - tf.expand_dims(tensor_z, 0)
        norm_weights_matrix = (
                tf.reshape(norm_weights, (-1, 1))
                * tf.reshape(norm_weights, (1, -1))
        )
        sample_cost = norm_squared(z_matrix) / (4 * gamma)
        sample_cost = phi_d(sample_cost, D) * norm_weights_matrix
        sample_cost = tf.reduce_sum(sample_cost)
        sample_cost *= 1 / tf.sqrt(4 * pi * gamma)

        dist_cost = norm_squared(tensor_z) / (2 + 4 * gamma)
        dist_cost = phi_d(dist_cost, D) * norm_weights
        dist_cost /= tf.sqrt(2 * pi * (1 + 2 * gamma))
        dist_cost = 2 * tf.reduce_sum(dist_cost)

        self_cost = 1 / tf.sqrt(2 * pi * (2 + 2 * gamma))

        return tf.log(tf.reduce_mean(sample_cost + self_cost - dist_cost))


    def gmm_imitation_cost(
            self, tensor_z, tensor_target,
            means, variances, probs, dataset, training_mode):

        D = tf.cast(tf.shape(tensor_z)[-1], tf.float32)
        N0 = tf.cast(tf.shape(tensor_z)[0], tf.float32)
        gamma = tf.pow(4 / (3 * N0 / dataset.classes_num), 0.4) * self.gamma_weight

        # shape: [data_points, class_num]
        kde_weights = tf.transpose(tensor_target / tf.reduce_sum(tensor_target, 0), [1, 0])
        # shape: [class_num, data_points]

        self_cost_tensor = 1 / tf.sqrt(2 * pi * (2 * variances + 2 * gamma))
        means = tf.expand_dims(means, 1)
        variances = tf.expand_dims(variances, 1)

        print("self", self_cost_tensor)
        dist_cost_tensor =  []
        sample_cost_tensor = []
        for idx in range(dataset.classes_num):
            # get indices with this label

            # TODO: nie wiem czy zadziala przy normalizacji (diagonala ok)
            # TODO: pomysl jeszcze nad tym
            z_matrix = tf.expand_dims(tensor_z, 1) - tf.expand_dims(tensor_z, 0)
            # [examples_num, examples_num, D]
            kde_weights_matrix = (
                    tf.reshape(kde_weights[idx], (1, 1, -1))
                    * tf.reshape(kde_weights, (dataset.classes_num, -1, 1))
            )
            # [class_num, data_points]
            sample_cost = norm_squared(z_matrix) / (4 * gamma)
            sample_cost = phi_d(sample_cost, D) * kde_weights_matrix
            sample_cost = tf.reduce_sum(sample_cost, axis=[1, 2])
            sample_cost *= 1 / tf.sqrt(4 * pi * gamma)

            sample_cost_tensor += [sample_cost]


            dist_cost = norm_squared(
                    tf.expand_dims(tensor_z, 0) - means) / (2 * variances + 4 * gamma)
            # shape: [class_num, data_points]
            # dist_cost = tf.Print(dist_cost, [tf.shape(dist_cost), tf.shape(tensor_z), tf.shape(means[idx])])
            # kde_weights[idx]
            dist_cost = phi_d(dist_cost, D) * kde_weights[idx]
            dist_cost /= tf.sqrt(2 * pi * (variances + 2 * gamma))
            dist_cost = 2 * tf.reduce_sum(dist_cost, axis=1)
            dist_cost_tensor += [dist_cost]



        sample_cost_tensor = tf.stack(sample_cost_tensor)
        dist_cost_tensor = tf.stack(dist_cost_tensor)

        logits = (sample_cost_tensor + self_cost_tensor - dist_cost_tensor)
        # logits = tf.Print(logits, [tf.shape(logits), tf.shape(sample_cost_tensor), tf.shape(dist_cost_tensor)])
        # logits powinien miec wymiary [classes_num]
        logits = -tf.log(logits)
        # logits = 1 / logits

        labels = tf.eye(dataset.classes_num)

        class_normalization = False
        if class_normalization:
            cwae_sum = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.eye(dataset.classes_num), logits=logits)
            cwae_sum *= probs
        else:
            cwae_sum = -tf.reduce_mean(tf.diag_part(logits))


        return tf.reduce_sum(cwae_sum)


    def ceclike_class_cost(self,
            tensor_z_all, tensor_target, labeled_mask,
            class_logits, means, variances, probs, dataset, training_mode):

        D = tf.cast(tf.shape(tensor_z_all)[-1], tf.float32)

        bool_labeled_mask = tf.cast(labeled_mask, tf.bool)
        tensor_target = tf.boolean_mask(tensor_target, bool_labeled_mask)
        labeled_tensor_z = tf.boolean_mask(tensor_z_all, bool_labeled_mask)
        labeled_argmaxed_tensor = tf.argmax(tensor_target, axis=-1)

        unlabeled_tensor_z = tf.boolean_mask(
            tensor_z_all, tf.logical_not(bool_labeled_mask))
        class_logits = tf.boolean_mask(
            class_logits, tf.logical_not(bool_labeled_mask))
        unlabeled_argmaxed_tensor = tf.argmax(class_logits, axis=-1)

        N0 = tf.cast(tf.shape(tensor_z_all)[0], tf.float32)
        gamma = tf.pow(4 / (3 * N0 / dataset.classes_num), 0.4)

        self_cost_tensor = 1 / tf.sqrt(2 * pi * (2 * variances + 2 * gamma))
        means = tf.expand_dims(means, 1)
        variances = tf.expand_dims(variances, 1)

        dist_cost_tensor =  []
        sample_cost_tensor = []
        for idx in range(dataset.classes_num):
            # get indices with this label
            mask = tf.equal(labeled_argmaxed_tensor, idx)
            labeled_class_z = tf.boolean_mask(labeled_tensor_z, mask)

            mask = tf.equal(unlabeled_argmaxed_tensor, idx)
            unlabeled_class_z = tf.boolean_mask(unlabeled_tensor_z, mask)
            all_class_z = tf.concat(
                [labeled_class_z, unlabeled_class_z], axis=0)
            # all_class_z = labeled_class_z

            z_matrix = tf.expand_dims(all_class_z, 1) - tf.expand_dims(all_class_z, 0)
            sample_cost = norm_squared(z_matrix) / (4 * gamma)
            sample_cost = tf.reduce_mean(phi_d(sample_cost, D))
            sample_cost *= 1 / tf.sqrt(4 * pi * gamma)

            sample_cost_tensor += [sample_cost]


            dist_cost = norm_squared(
                tf.expand_dims(all_class_z, 0) - means) / (2 * variances + 4 * gamma)
            dist_cost = phi_d(dist_cost, D)
            dist_cost /= tf.sqrt(2 * pi * (variances + 2 * gamma))
            dist_cost = tf.reduce_mean(dist_cost, axis=-1)
            dist_cost *= 2 * 1
            dist_cost_tensor += [dist_cost]


        sample_cost_tensor = tf.stack(sample_cost_tensor)
        dist_cost_tensor = tf.stack(dist_cost_tensor)

        sample_cost_tensor = tf.expand_dims(sample_cost_tensor, 1)
        self_cost_tensor = tf.expand_dims(self_cost_tensor, 0)
        logits = (sample_cost_tensor + self_cost_tensor - dist_cost_tensor)
        logits = -tf.log(logits)
        # logits = 1 / logits

        labels = tf.eye(dataset.classes_num)
        labeled_num = tf.reduce_sum(tf.cast(labeled_mask, tf.float32))

        class_normalization = True
        if class_normalization:
            cwae_sum = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.eye(dataset.classes_num), logits=logits)
            cwae_sum *= probs
        else:
            cwae_sum = tf.cond(
                tf.greater(labeled_num, 0),
                lambda: -1 * tf.diag_part(logits) * probs,
                lambda: tf.zeros([], tf.float32))


        return tf.reduce_sum(cwae_sum)

    def cwae_class_cost(self, tensor_z_all, tensor_target, labeled_mask,
                        means, variances, probs, dataset, gamma):

        D = tf.cast(tf.shape(tensor_z_all)[-1], tf.float32)
        G = tf.cast(tf.shape(means)[0], tf.float32)

        cwae_sum = tf.zeros([], dtype=tf.float32)
        tensor_z = tf.boolean_mask(tensor_z_all, labeled_mask)
        tensor_target = tf.boolean_mask(tensor_target, labeled_mask)
        argmaxed_tensor = tf.argmax(tensor_target, axis=-1)

        # N0 = 100 # ???
        # gamma = tf.pow(4 / (3 * N0 / G), 0.4)
        sample_cost_tensor = []
        # probs = tf.ones_like(probs)

        self_cost_tensor = probs * probs / tf.sqrt(2 * pi * (2 * variances + 2 * gamma))
        # self_cost_tensor = tf.Print(self_cost_tensor, [probs * probs])
        dist_cost_tensor =  []
        means = tf.expand_dims(means, 1)
        variances = tf.expand_dims(variances, 1)
        classes_props = np.sum(tensor_target, 0)

        for idx in range(dataset.classes_num):
            # get indices with this label
            mask = tf.equal(argmaxed_tensor, idx)
            labeled_z = tf.boolean_mask(tensor_z, mask)

            z_matrix = tf.expand_dims(labeled_z, 1) - tf.expand_dims(labeled_z, 0)
            sample_cost = norm_squared(z_matrix) / (4 * gamma)
            sample_cost = tf.reduce_mean(phi_d(sample_cost, D))
            sample_cost *= probs[idx] * probs[idx] / tf.sqrt(4 * pi * gamma)

            sample_cost_tensor += [sample_cost]


            dist_cost = norm_squared(
                tf.expand_dims(labeled_z, 0) - means) / (2 * variances + 4 * gamma)
            dist_cost = phi_d(dist_cost, D)
            dist_cost /= tf.sqrt(2 * pi * (variances + 2 * gamma))
            dist_cost = tf.reduce_mean(dist_cost, axis=-1)
            dist_cost *= 2 * probs[idx] * probs
            dist_cost_tensor += [dist_cost]


        sample_cost_tensor = tf.stack(sample_cost_tensor)
        dist_cost_tensor = tf.stack(dist_cost_tensor)

        sample_cost_tensor = tf.expand_dims(sample_cost_tensor, 1)
        self_cost_tensor = tf.expand_dims(self_cost_tensor, 0)
        logits = (sample_cost_tensor + self_cost_tensor - dist_cost_tensor)
        logits = -tf.log(logits)
        labels = tf.eye(dataset.classes_num)
        labeled_num = tf.reduce_sum(tf.cast(labeled_mask, tf.float32))

        # Normalize
        normalize = True
        if normalize:

            cwae_sum = tf.cond(
                tf.greater(labeled_num, 0),
                lambda: tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.eye(dataset.classes_num), logits=logits)
                    * classes_props / labeled_num,
                lambda: tf.zeros([], tf.float32))
        else:
            cwae_sum = tf.cond(
                tf.greater(labeled_num, 0),
                lambda: tf.diag_part(logits) * classes_props / labeled_num,
                lambda: tf.zeros([], tf.float32))

        return tf.reduce_sum(cwae_sum)

class MultiGmmCwaeModel():
    def __init__(
        self, name, coder, dataset, latent_dim=300, gaussians_per_class=1,
        learning_rate=1e-3, gamma_weight=1.0, cw_weight=1.0,
        supervised_weight=1.0, init=1.0):


        self.name = name
        self.z_dim = latent_dim
        self.gaussians_per_class = gaussians_per_class
        tf.reset_default_graph()


        optimizer = tft.AdamOptimizer(learning_rate)
        tensor_x = tf.placeholder(
                shape=[None, dataset.x_dim],
                dtype=tf.float32, name='input_x')
        tensor_labels = tf.placeholder(
                shape=[None, dataset.classes_num],
                dtype=tf.float32, name='target_y')
        train_labeled = tf.placeholder_with_default(True, shape=[])
        tensor_training = tf.placeholder_with_default(False, shape=[])

        print("gaussians_per_class", gaussians_per_class)
        gaussians_num = dataset.classes_num * gaussians_per_class
        labeled_mask = get_labels_mask(tensor_labels)
        means, variances, probs = get_gaussians(
            latent_dim, init, dataset, gaussians_num)

        tensor_z = coder.encode(tensor_x, latent_dim, tensor_training)
        tensor_y = coder.decode(tensor_z, dataset.x_dim, tensor_training)

        rec_cost = norm_squared(tensor_x - tensor_y, axis=-1)
        rec_cost = tf.cond(
            train_labeled,
            lambda: tf.reduce_mean(rec_cost),
            lambda: tf.reduce_mean(
                tf.boolean_mask(rec_cost, tf.logical_not(labeled_mask)))
        )

        unsupervised_tensor_z = tf.cond(
            train_labeled,
            lambda: tensor_z,
            lambda: tf.boolean_mask(tensor_z, tf.logical_not(labeled_mask))
        )

        N0 = tf.shape(unsupervised_tensor_z)[0]
        gamma = tf.cast(tf.pow(4 / (3 * N0 / gaussians_num), 0.4), tf.float32)
        gamma *= gamma_weight

        # TODO: clean this mess up
        class_logits = calculate_logits(tensor_z, means, variances, probs)
        class_probs = tf.nn.softmax(class_logits) # [N, G * K]
        class_probs = tf.reshape(class_probs, (-1, dataset.classes_num, gaussians_per_class))
        class_probs = tf.reduce_sum(class_probs, -1)

        class_cost = calculate_probs_cost(
            class_probs, tensor_labels, labeled_mask)
        class_cost *= supervised_weight


        # TODO: log_cw
        cw_cost = cramer_wold_distance(unsupervised_tensor_z, means, variances, probs, gamma)
        log_cw_cost = cw_weight * tf.log(cw_cost)

        full_cost = tf.reduce_mean(rec_cost + log_cw_cost + class_cost)
        train_op = optimizer.minimize(full_cost)

        self.saver = tf.train.Saver(max_to_keep=10000)
        self.placeholders = {
            "X": tensor_x,
            "y": tensor_labels,
            "train_labeled": train_labeled,
            "training": tensor_training,
        }

        self.costs = {
            "full": full_cost,
            "class": class_cost,
            "reconstruction": rec_cost,
            "cw": log_cw_cost
        }

        self.train_ops = {
            "full": train_op,
        }

        self.gausses = {
            "means": means,
            "variations": variances,
            "probs": probs
        }
        self.out = {
            "z": tensor_z,
            "y": tensor_y,
            "logits": class_logits,
            "probs": class_probs
        }
        self.preds = class_probs

# ======================================
# ========== HELPER FUNCTIONS ==========
# ======================================


def get_labels_mask(tensor_labels):
    one = tf.constant(1, tf.float32)
    labels_mask = tf.equal(one, tf.reduce_sum(tensor_labels, axis=-1))
    labels_mask = tf.cast(labels_mask, tf.bool)
    return labels_mask

def norm_squared(X, axis=-1):
    return tf.reduce_sum(tf.square(X), axis=axis)


def norm(X, axis=-1):
    return tf.norm(tf.square(X), axis=axis)


def phi_f(s):
    t = s/7.5
    return tf.exp(-s/2) * (
            1 + 3.5156229*t**2 + 3.0899424*t**4 + 1.2067492*t**6
            + 0.2659732*t**8 + 0.0360768*t**10 + 0.0045813*t**12)


def phi_g(s):
    t = s/7.5
    return tf.sqrt(2/s) * (
            0.39894228 + 0.01328592*t**(-1) + 0.00225319*t**(-2)
            - 0.00157565*t**(-3) + 0.0091628*t**(-4) - 0.02057706*t**(-5)
            + 0.02635537*t**(-6) - 0.01647633*t**(-7) + 0.00392377*t**(-8))


def phi(x):
    a = 7.5
    return phi_f(tf.minimum(x, a)) - phi_f(a) + phi_g(tf.maximum(x, a))


def phi_d(s, D):
    if D == 2:
        return phi(s)
    else:
        return 1 / tf.sqrt(1 + (4 * s) / (2 * D - 3))


def get_gauss_std(tensor_z, means, variances, probs, classes_num):
    likelihood = -norm_squared(
        tf.expand_dims(tensor_z, 1) - tf.expand_dims(means, 0))
    likelihood /= 2 * variances
    likelihood -= 1/2 * tf.log(variances)
    clustering = tf.argmax(likelihood, 1)
    z_vars = []
    for idx in range(classes_num):
        indices = tf.where(tf.equal(clustering, idx))
        class_z = tf.gather(tensor_z, indices)
        _, z_var = tf.nn.moments(class_z, axes=[0])
        z_var = tf.where(tf.is_nan(z_var), tf.ones_like(z_var), z_var)
        z_vars += [tf.reduce_mean(z_var)]

    z_std = tf.sqrt(tf.reduce_mean(tf.stack(z_vars)))
    return z_std


def get_global_std(tensor_z, means, variances, probs, classes_num):
    _, z_var = tf.nn.moments(tensor_z, axes=[0])
    z_std = tf.sqrt(tf.reduce_mean(z_var))
    return z_std


def get_gaussians(z_dim, init, dataset, gauss_num):
    G = gauss_num
    with tf.variable_scope("gmm", reuse=False):
        one_hot = np.zeros([G, z_dim])
        one_hot[np.arange(z_dim) % G, np.arange(z_dim)] = 1
        one_hot *= init / z_dim * G
        one_hot += np.random.normal(0, .001, size=one_hot.shape)
        # TODO: means = tf.constant()


        variable_means = True
        if gauss_num == 1:
            means = tf.constant([[0] * z_dim], dtype=tf.float32)
        elif variable_means:
            # means_initialization = tf.constant_initializer(
            #     np.random.normal(0, init, size=(G, z_dim)))
            means_initialization = tf.constant_initializer(one_hot)
            means = tf.get_variable(
                    "gmm_means", [G, z_dim],
                    initializer=means_initialization)
        else:
            means = tf.constant(one_hot, dtype=tf.float32)
                # initializer=tf.random_uniform_initializer(-1.0, 1.0))



        # var_initialization = tf.constant_initializer(
        #     np.random.uniform(0.0, 1.0, size=(G)))
        # beta = tf.get_variable("gmm_betas", [G], initializer=var_initialization)
        beta = tf.zeros([G], name="betas")
        variances = 1.0 + tf.abs(beta)

        # Three ways to calculate p_k

        labels_proportions = (
            dataset.labeled_train["y"].sum(0) / dataset.labeled_train["y"].sum())
        probs = tf.constant(labels_proportions, dtype=tf.float32)
        gaussians_per_class = gauss_num // len(labels_proportions)
        probs = tf.tile(tf.expand_dims(probs, -1), (1, gaussians_per_class))
        probs = tf.reshape(probs, (-1,))

        # probs = tf.constant([1 / G] * G, dtype=tf.float32)

        # logits = tf.get_variable("logit_probs", [G])
        # probs = tf.nn.softmax(logits)

    print("Shape of gaussians", means.shape, gauss_num)
    return means, variances, probs

def get_empirical_gamma(tensor_z, means, variances, probs, classes_num):
    N0 = tf.cast(tf.shape(tensor_z)[0], tf.float32)
    gauss_std = get_gauss_std(tensor_z, means, variances, probs, classes_num)
    global_std = get_global_std(tensor_z, means, variances, probs, classes_num)
    default_gamma = tf.pow(4 * tf.pow(gauss_std, 5) / (3 * N0 / classes_num), 0.4)
    return tf.stop_gradient(default_gamma)

# =========================================
# ========== VARIOUS CLASSIFIERS ==========
# =========================================

def calculate_cw_logits(tensor_z, means, alpha, p, gamma=None):
    N = tf.cast(tf.shape(tensor_z)[0], tf.float32)
    D = tf.cast(tf.shape(tensor_z)[1], tf.float32)
    # N0 = 100
    N0 = N // 2 # TODO: czy nie?
    G = tf.shape(means)[0]

    if gamma is None:
        gamma = tf.pow(4 / (3 * (N0 / G)), 0.4)
        # gamma = tf.pow(4 / (3 / G), 0.4)

    m_X_sub_matrix = tf.subtract(tf.expand_dims(means, 0), tf.expand_dims(tensor_z, 1))
    C1 = norm_squared(m_X_sub_matrix, axis=2)
    C2 = phi_d(C1 / (2 * (alpha + 2 * gamma)), D)
    C = 2 * p * 1 / G / tf.sqrt(2 * pi * (alpha + 2 * gamma)) * C2
    A = (1 / (G * G)) / tf.sqrt(4 * pi * gamma)
    B = tf.square(p) / tf.sqrt(2 * pi * (2 * alpha + 2 * gamma))

    C3 = C2 # + tf.log(p) / 100
    print(C3.shape)
    return -tf.log(A + B - C)

def calculate_cw_probs(tensor_z, means, alpha, p):
    logits = calculate_cw_logits(tensor_z, means, alpha, p)
    probs = logits / tf.expand_dims(tf.reduce_sum(logits, axis=1), 1)
    print(probs.shape)
    return probs
    # C = tf.reduce_sum(C3)

def calculate_pdf_logits(tensor_z, means, alpha, p):
    m_X_sub_matrix = tf.subtract(tf.expand_dims(means, 0), tf.expand_dims(tensor_z, 1))
    pdf = tf.exp(-norm_squared(m_X_sub_matrix) / (1e6 * 2 * alpha))
    pdf /= tf.sqrt(2 * pi * alpha)
    return pdf

def calculate_pdf_probs(tensor_z, means, alpha, p):
    logits = calculate_pdf_logits(tensor_z, means, alpha, p)
    probs = logits / tf.expand_dims(tf.reduce_sum(logits, axis=1), 1)
    print(probs.shape)
    return probs

def calculate_softmax_logits(tensor_z, means, alpha, p):
    class_logits = -norm_squared(tf.expand_dims(tensor_z, 1) - tf.expand_dims(means, 0), axis=-1) / (2 * alpha)
    # class_logits = tf.log(p) + -1/2 * tf.log(2 * pi * alpha) + class_logits
    class_logits = tf.nn.softmax(class_logits)
    return class_logits

def calculate_logits(tensor_z, means, alpha, p):
    class_logits = -norm_squared(tf.expand_dims(tensor_z, 1) - tf.expand_dims(means, 0), axis=-1) / (2 * alpha)
    class_logits = tf.log(p) - 0.5 * tf.log(2 * pi * alpha) + class_logits
    return class_logits

# TODO: to nie dziala chyba. Poprawic.
def calculate_multiple_logits(tensor_z, means, alpha, p, gaussians_per_class=1):
    N = tf.cast(tf.shape(tensor_z)[0], tf.int32)
    class_logits = -norm_squared(tf.expand_dims(tensor_z, 1) - tf.expand_dims(means, 0), axis=-1) / (2 * alpha)
    class_logits = tf.log(p) - 0.5 * tf.log(2 * pi * alpha) + class_logits # [N, G * K]
    class_logits = tf.reshape(class_logits, [N, -1, gaussians_per_class]) # [N, G, K]
    class_logits = tf.reduce_sum(class_logits, -1) # [N, G]
    return class_logits

def calculate_single_class_probs(tensor_z, means, alpha, p):
    class_probs = -norm_squared(tf.expand_dims(tensor_z, 1) - tf.expand_dims(means, 0), axis=-1) / (2 * alpha)
    # class_probs *= tf.sqrt(2 * pi * alpha)
    # class_probs /= tf.sqrt(2 * pi * alpha)  # Normalization
    return class_probs

def single_class_cross_entropy(class_probs, tensor_target, labeled_mask):
    # class_probs: (N x G)
    # cross_entropy = -tf.reduce_sum(tensor_target * tf.log(class_probs + 1e-3), axis=-1)
    cross_entropy = tf.reduce_sum(tf.exp(-class_probs) * tensor_target, axis=-1)
    denominator = tf.reduce_sum(tensor_target)
    denominator = tf.cond(tf.equal(denominator, 0), lambda: tf.constant(1, tf.float32), lambda: denominator)
    # class_cost_fin = tf.reduce_mean(class_cost * non_zero_indices)
    class_cost_fin = tf.reduce_sum(cross_entropy * labeled_mask) / denominator
    return class_cost_fin

def calculate_probs_cost(
    class_probs, tensor_target,
    labeled_mask, eps=0):

    print(class_probs.shape, tensor_target.shape)
    class_cost = tf.reduce_mean(
        -tf.reduce_sum(tensor_target * tf.log(tf.math.maximum(class_probs, 1e-15)), reduction_indices=[1]))

    class_cost = tf.where(
        tf.less(class_cost, eps),
        tf.zeros_like(class_cost),
        class_cost)
    denominator = tf.reduce_sum(tensor_target)
    denominator = tf.cond(tf.equal(denominator, 0), lambda: tf.constant(1, tf.float32), lambda: denominator)
    # class_cost_fin = tf.reduce_mean(class_cost * non_zero_indices)
    class_cost_fin = tf.reduce_sum(class_cost * tf.cast(labeled_mask, tf.float32)) / denominator
    return class_cost_fin

def calculate_logits_cost(
    class_logits, tensor_target,
    labeled_mask, eps=0):
    class_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=class_logits, labels=tensor_target)
    # class_cost = tf.where(
    #     tf.less(class_cost, eps),
    #     tf.zeros_like(class_cost),
    #     class_cost)
    denominator = tf.reduce_sum(tensor_target)
    denominator = tf.cond(tf.equal(denominator, 0), lambda: tf.constant(1, tf.float32), lambda: denominator)
    # class_cost_fin = tf.reduce_mean(class_cost * non_zero_indices)

    class_cost_fin = tf.reduce_sum(class_cost * tf.cast(labeled_mask, tf.float32)) / denominator
    return class_cost_fin

def calculate_logits_entropy(class_logits, labeled_mask):

    tensor_target = tf.nn.softmax(class_logits)
    entropy_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=class_logits, labels=tensor_target)

    denominator = tf.reduce_sum(tf.cast(tf.logical_not(labeled_mask), tf.float32))
    denominator = tf.cond(tf.equal(denominator, 0), lambda: tf.constant(1, tf.float32), lambda: denominator)
    # entropy_cost_fin = tf.reduce_mean(entropy_cost * non_zero_indices)

    entropy_cost_fin = tf.reduce_sum(entropy_cost * (1 - tf.cast(labeled_mask, tf.float32))) / denominator
    return entropy_cost_fin




# =================================================
# ========== VARIOUS PUSH-AWAY FUNCTIONS ==========
# =================================================

def entropy_regularization(class_logits):
    class_probs = tf.nn.softmax(class_logits, axis=-1)
    self_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=class_probs, logits=class_logits)
    return tf.reduce_mean(self_entropy)

def dkl_distance_penalty(z_dim, means, variances, classes_num):
    ratio = tf.expand_dims(variances, 1) / tf.expand_dims(variances, 0)
    cost = tf.reduce_sum(tf.square(tf.expand_dims(means, 1) - tf.expand_dims(means, 0)), axis=-1)
    cost /= tf.transpose(variances)
    cost = z_dim * ratio + cost - z_dim + z_dim * tf.log(1 / ratio)
    cost = 1/2 * cost
    mask = tf.ones(classes_num) - tf.eye(classes_num)
    cost = tf.minimum(1e6, cost)
    return -tf.reduce_mean(tf.log(cost + 1e-4) * mask)

def truncated_distance_penalty(z_dim, means, variances, classes_num):
    dist = 1e4 * (tf.expand_dims(variances, 1) + tf.expand_dims(variances, 0))
    dist -= tf.reduce_sum(tf.square(tf.expand_dims(means, 1) - tf.expand_dims(means, 0)), axis=-1)
    mask = tf.ones(classes_num) - tf.eye(classes_num)
    cost = tf.reduce_mean(tf.maximum(0.0, dist) * mask)
    return cost

def linear_distance_penalty(z_dim, means, variances, probs, classes_num):
    dist = tf.reduce_sum(tf.square(tf.expand_dims(means, 1) - tf.expand_dims(means, 0)), axis=-1)
    dist /= variances # (tf.expand_dims(variances, 1) + tf.transpose(variances))
    mask = tf.ones([classes_num, classes_num]) - tf.eye(classes_num)
    dist = tf.boolean_mask(dist, tf.cast(mask, tf.bool))
    return -tf.reduce_mean(tf.log(tf.sqrt(dist))) # sqrt

def closest_distance_penalty(z_dim, means, variances, probs, classes_num):
    dist = tf.reduce_sum(tf.square(tf.expand_dims(means, 1) - tf.expand_dims(means, 0)), axis=-1)
    dist /= tf.transpose(variances)
    mask = tf.ones([classes_num, classes_num]) - tf.eye(classes_num)
    dist = tf.boolean_mask(dist, tf.cast(mask, tf.bool))
    return -tf.reduce_mean(tf.reduce_min(dist, axis=0))

def min_distance_penalty(z_dim, means, variances, classes_num):
    dist = tf.reduce_sum(tf.square(tf.expand_dims(means, 1) - tf.expand_dims(means, 0)), axis=-1)
    dist /= tf.transpose(variances)
    mask = tf.ones([classes_num, classes_num]) - tf.eye(classes_num)
    dist = tf.boolean_mask(dist, tf.cast(mask, tf.bool))
    return -tf.reduce_mean(tf.sqrt(tf.reduce_min(dist))) # sqrt

def cw_distance_penalty(self, z_dim, means, variances, probs, classes_num):
    # probs *= 10

    N0 = 100
    G = tf.shape(means)[0]
    # gamma = tf.pow(4 / (3 * N0 / G), 0.4)
    gamma = self.gamma
    # gamma = 5.0 # evil trick
    var_arr = tf.expand_dims(variances, 0) + tf.expand_dims(variances, 1)
    p_arr = tf.expand_dims(probs, 1) * tf.expand_dims(probs, 0)
    square_probs = tf.square(probs)
    # probs = np.ones_like(probs)

    means_arr = norm_squared(tf.expand_dims(means, 0) - tf.expand_dims(means, 1))
    A = tf.expand_dims(square_probs, 0) / tf.sqrt(2 * pi * (2 * tf.expand_dims(variances, 0) + gamma))
    B = tf.expand_dims(square_probs, 1) / tf.sqrt(2 * pi * (2 * tf.expand_dims(variances, 1) + gamma))
    C = phi_d(means_arr / (2 * (var_arr + gamma)), z_dim)
    C = 2 * p_arr / tf.sqrt(2 * pi * (var_arr + gamma)) * C
    dist = A + B - C

    mask = tf.ones([classes_num, classes_num]) - tf.eye(classes_num)
    dist = tf.boolean_mask(dist, tf.cast(mask, tf.bool))
    dist = tf.reshape(dist, [classes_num, classes_num - 1])
    return -tf.reduce_mean(tf.log(dist))

def mincw_distance_penalty(self, z_dim, means, variances, classes_num):
    n = 1
    # gamma = tf.pow(4 / (3 * n), 0.4) * n / 10
    gamma = self.gamma
    var_arr = tf.expand_dims(variances, 0) + tf.expand_dims(variances, 1)
    means_arr = norm_squared(tf.expand_dims(means, 0) - tf.expand_dims(means, 1))
    dist = phi_d(means_arr / (2 * (var_arr + 2 * gamma)), z_dim)
    mask = tf.ones([classes_num, classes_num]) - tf.eye(classes_num)
    dist = tf.boolean_mask(dist, tf.cast(mask, tf.bool))
    return tf.log(tf.reduce_max(dist))
