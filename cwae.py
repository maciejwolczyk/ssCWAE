import numpy as np
import tensorflow as tf
import tensorflow.train as tft
from math import pi


def cramer_wold_distance(X, m, variance, p, gamma):
    N = tf.cast(tf.shape(X)[0], tf.float32)
    D = tf.cast(tf.shape(X)[1], tf.float32)

    variance_matrix = tf.expand_dims(variance, 0) + tf.expand_dims(variance, 1)
    X_sub_matrix = tf.subtract(tf.expand_dims(X, 0), tf.expand_dims(X, 1))
    A1 = norm_squared(X_sub_matrix, axis=2)

    A1 = tf.reduce_sum(phi_d(A1 / (4 * gamma), D))
    A = 1/(N*N * tf.sqrt(2 * pi * 2 * gamma)) * A1

    m_sub_matrix = tf.subtract(tf.expand_dims(m, 0), tf.expand_dims(m, 1))
    p_mul_matrix = tf.matmul(tf.expand_dims(p, 1), tf.expand_dims(p, 0))
    B1 = norm_squared(m_sub_matrix, axis=2)
    B2 = phi_d(B1 / (2 * variance_matrix + 4 * gamma), D)
    B3 = p_mul_matrix / tf.sqrt(2 * pi * (variance_matrix + 2 * gamma))
    B = tf.reduce_sum(B3 * B2)

    m_X_sub_matrix = tf.subtract(tf.expand_dims(m, 0), tf.expand_dims(X, 1))
    C1 = norm_squared(m_X_sub_matrix, axis=2)
    C2 = phi_d(C1 / (2 * (variance + 2 * gamma)), D)
    C3 = 2 * p / (N * tf.sqrt(2 * pi * (variance + 2 * gamma))) * C2
    C = tf.reduce_sum(C3)
    return tf.reduce_mean(A + B - C)


class GmmCwaeModel():
    def __init__(
            self, name, coder, dataset,
            z_dim=300, supervised_weight=1.0, distance_weight=1.0,
            learning_rate=1e-3, cw_weight=1.0, init=1.0):

        tf.reset_default_graph()
        self.name = name
        self.init = init
        self.optimizer = tft.AdamOptimizer(learning_rate)
        self.cw_weight = cw_weight

        self.z_dim = z_dim
        x_dim = dataset.x_dim

        # Prepare placeholders
        tensor_x = tf.placeholder(
                shape=[None, x_dim],
                dtype=tf.float32, name='input_x')
        tensor_labels = tf.placeholder(
                shape=[None, dataset.classes_num],
                dtype=tf.float32, name='target_y')

        train_labeled = tf.placeholder_with_default(True, shape=[])
        tensor_cw_weight = tf.placeholder_with_default(cw_weight, shape=[])
        tensor_training = tf.placeholder_with_default(False, shape=[])

        labeled_mask = get_labels_mask(tensor_labels)
        tensor_z = coder.encode(tensor_x, z_dim, tensor_training)
        tensor_y = coder.decode(tensor_z, x_dim, tensor_training)

        # Unsupervised examples are treated differently than supervised:
        unsupervised_tensor_z = tf.cond(
            train_labeled,
            lambda: tensor_z,
            lambda: tf.boolean_mask(tensor_z, tf.logical_not(labeled_mask)))
        N0 = tf.shape(unsupervised_tensor_z)[0]

        means, variances, probs = get_gaussians(
                z_dim, init, dataset, dataset.classes_num)

        gamma = tf.pow(4 / (3 * N0 / dataset.classes_num), 0.4)
        gamma = tf.cast(gamma, tf.float32)

        class_logits = calculate_logits(
            tensor_z, means, variances, probs)
        class_probs = tf.nn.softmax(class_logits)
        class_cost = calculate_logits_cost(
                class_logits, tensor_labels, labeled_mask)

        cw_cost = cramer_wold_distance(
                unsupervised_tensor_z, means, variances, probs, gamma)
        log_cw_cost = tf.log(cw_cost)
        log_cw_cost *= tensor_cw_weight

        # MSE
        rec_cost = norm_squared(tensor_x - tensor_y, axis=-1)
        rec_cost = tf.cond(
            train_labeled,
            lambda: tf.reduce_mean(rec_cost),
            lambda: tf.reduce_mean(
                tf.boolean_mask(rec_cost, tf.logical_not(labeled_mask)))
        )

        distance_cost = linear_distance_penalty(
                z_dim, means, variances, probs, dataset.classes_num)

        unsupervised_cost = tf.reduce_mean(
                rec_cost
                + log_cw_cost
                + distance_weight * distance_cost)

        full_cost = tf.reduce_mean(
                rec_cost
                + log_cw_cost
                + supervised_weight * class_cost
                + distance_weight * distance_cost
                )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Prepare various train ops
            grads, gvars = zip(*self.optimizer.compute_gradients(full_cost))
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            capped_gvs = [
                    (tf.clip_by_value(grad, -1., 1.), var)
                    for grad, var in zip(grads, gvars)
                ]
            train_op = self.optimizer.apply_gradients(capped_gvs)

            class_train_op = self.optimizer.minimize(class_cost)
            rec_train_op = self.optimizer.minimize(rec_cost)
            cw_train_op = self.optimizer.minimize(log_cw_cost)
            supervised_train_op = self.optimizer.minimize(class_cost)

        # Prepare variables for outside use
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.placeholders = {
            "X": tensor_x,
            "y": tensor_labels,
            "train_labeled": train_labeled,
            "cw_weight": tensor_cw_weight,
            "training": tensor_training,
        }

        self.out = {
            "logits": class_logits,
            "probs": class_probs,
            "z": tensor_z,
            "y": tensor_y,
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
            "full": full_cost,
            "unsupervised": unsupervised_cost,
        }

        self.train_ops = {
            "full": train_op,
            "supervised": supervised_train_op,
            "rec": rec_train_op,
            "class": class_train_op,
            "cw": cw_train_op,
        }

        self.train_op = train_op
        self.supervised_train_op = supervised_train_op
        self.preds = class_logits


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


def get_gaussians(z_dim, init, dataset, gauss_num):
    G = gauss_num
    with tf.variable_scope("gmm", reuse=False):

        if G <= z_dim:
            one_hot = np.zeros([G, z_dim])
            one_hot[np.arange(z_dim) % G, np.arange(z_dim)] = 1
            one_hot *= init / z_dim * G
            one_hot += np.random.normal(0, .001, size=one_hot.shape)
        else:
            one_hot = np.random.normal(0, init, size=(G, z_dim))

        if gauss_num == 1:
            initializer = tf.constant_initializer([[0] * z_dim])
            means = tf.get_variable(
                "gmm_means", [1, z_dim],
                initializer=initializer, dtype=tf.float32
            )
        else:
            means_initialization = tf.constant_initializer(one_hot)
            means = tf.get_variable(
                    "gmm_means", [G, z_dim],
                    initializer=means_initialization)

        beta = tf.zeros([G], name="betas")
        variances = 1.0 + tf.abs(beta)

        labels_proportions = (
            dataset.labeled_train["y"].sum(0)
            / dataset.labeled_train["y"].sum()
            )
        probs = tf.constant(labels_proportions, dtype=tf.float32)
        probs = tf.reshape(probs, (-1,))

        if gauss_num == 1:
            probs = tf.constant([1.], dtype=tf.float32)
    print("Shape of gaussians", means.shape, gauss_num)
    return means, variances, probs


# =========================================
# ========== VARIOUS CLASSIFIERS ==========
# =========================================


def calculate_logits(tensor_z, means, variance, p):
    D = tf.cast(tf.shape(means)[-1], tf.float32)
    diffs = tf.expand_dims(tensor_z, 1) - tf.expand_dims(means, 0)
    class_logits = -norm_squared(diffs, axis=-1) / (2 * variance)
    class_logits = tf.log(p) - 0.5 * D * tf.log(2 * pi * variance) + class_logits
    return class_logits


def calculate_logits_cost(
        class_logits, tensor_target,
        labeled_mask, eps=0):
    class_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=class_logits, labels=tensor_target)
    denominator = tf.reduce_sum(tensor_target)
    denominator = tf.cond(
            tf.equal(denominator, 0),
            lambda: tf.constant(1, tf.float32),
            lambda: denominator
        )

    casted_mask = tf.cast(labeled_mask, tf.float32)
    class_cost_fin = tf.reduce_sum(class_cost * casted_mask) / denominator
    return class_cost_fin


# =================================================
# ========== VARIOUS PUSH-AWAY FUNCTIONS ==========
# =================================================

def linear_distance_penalty(z_dim, means, variances, probs, classes_num):
    diffs = tf.expand_dims(means, 1) - tf.expand_dims(means, 0)
    dist = tf.reduce_sum(tf.square(diffs), axis=-1)
    dist /= variances
    mask = tf.ones([classes_num, classes_num]) - tf.eye(classes_num)
    dist = tf.boolean_mask(dist, tf.cast(mask, tf.bool))
    return -tf.reduce_mean(tf.sqrt(dist))
