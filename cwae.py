import numpy as np
import tensorflow as tf
import tensorflow.train as tft
from math import pi

def cramer_wold_distance(X, m, alpha, p, gamma):
    N = tf.cast(tf.shape(X)[0], tf.float32)
    D = tf.cast(tf.shape(X)[1], tf.float32)
    # N0 = 73257
    N0 = N

    alpha_matrix = tf.expand_dims(alpha, 0) + tf.expand_dims(alpha, 1)

    # X_sub_matrix = tf.subtract(tf.expand_dims(X, 0), tf.expand_dims(X, 1))
    # A1 = norm_squared(X_sub_matrix, axis=2)
    r = tf.reduce_sum(X * X, -1)
    r = tf.reshape(r, [-1, 1])
    A1 = r - 2 * tf.matmul(X, tf.transpose(X)) + tf.transpose(r)

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



class CwaeModel():
    def __init__(
        self, name, coder, dataset, latent_dim=300,
        supervised_weight=1.0, distance_weight=1.0,
        optimizer=tft.AdamOptimizer(1e-3),
        eps=1e-2, init=1.0, gamma=1.0):

        tf.reset_default_graph()
        self.name = name
        self.init = init
        self.gamma = gamma
        gamma_weight = gamma
        self.coder = coder
        self.optimizer = optimizer

        im_h, im_w, im_c = dataset.image_shape
        x_dim = dataset.x_dim
        z_dim = latent_dim


        # Prepare placeholders
        tensor_x = tf.placeholder(
                shape=[None, x_dim],
                dtype=tf.float32, name='input_x')
        tensor_labels = tf.placeholder(
                shape=[None, dataset.classes_num],
                dtype=tf.float32, name='target_y')
        tensor_x_target = tf.placeholder(
                shape=[None, x_dim],
                dtype=tf.float32, name='target_input_x')
        train_labeled = tf.placeholder_with_default(True, shape=[])
        tensor_distance_weight = tf.placeholder_with_default(distance_weight, shape=[])
        tensor_supervised_weight = tf.placeholder_with_default(supervised_weight, shape=[])
        tensor_training = tf.placeholder_with_default(False, shape=[])
        labeled_mask = self.get_labels_mask(tensor_labels)

        # Coder and decoder
        tensor_z = coder.encode(tensor_x, z_dim, tensor_training)
        tensor_y = coder.decode(tensor_z, x_dim, tensor_training)

        # Unsupervised examples are treated differently than supervised:
        unsupervised_tensor_z = tf.cond(
            train_labeled,
            lambda: tensor_z,
            lambda: tf.boolean_mask(tensor_z, tf.cast(1 - labeled_mask, tf.bool)))

        means, variances, probs = get_gaussians(dataset.classes_num, z_dim, init, dataset)
        default_gamma = get_empirical_gamma(
                unsupervised_tensor_z, means, variances, probs, dataset.classes_num)
        tensor_gamma = tf.placeholder_with_default(default_gamma, shape=[])
        self.gamma = tensor_gamma * gamma_weight

        # Old Cramer-Wold cost
        cw_cost = cramer_wold_distance(
                unsupervised_tensor_z, means, variances, probs, gamma)
        log_cw_cost = tf.log(cw_cost)

        rec_cost = norm_squared(tensor_x_target - tensor_y, axis=-1)
        log_rec_cost = tf.cond(
            train_labeled,
            lambda: tf.reduce_mean(tf.log(rec_cost)),
            lambda: tf.reduce_mean(tf.log(rec_cost) * (1 - labeled_mask)))

        rec_cost = tf.cond(
            train_labeled,
            lambda: tf.reduce_mean(rec_cost),
            lambda: tf.reduce_mean(rec_cost * (1 - labeled_mask)))

        class_logits = calculate_logits(tensor_z, means, variances, probs)
        class_probs = tf.nn.softmax(class_logits)
        # class_logits = class_probs # nie do konca

        # class_probs = self.calculate_cw_probs(tensor_z, means, alpha, p)
        # class_cost = self.calculate_logits_cost(
        #   class_logits, tensor_labels, labeled_mask, eps=eps)

        class_cost = self.cwae_class_cost(
            tensor_z, tensor_labels, labeled_mask,
            means, variances, probs, dataset)

        # New Cramer-Wold cost
        cec_cost = self.ceclike_class_cost(
            tensor_z, tensor_labels, labeled_mask,
            class_logits, means, variances, probs, dataset)

        # class_logits = self.calculate_single_class_probs(
        #     tensor_z, means, alpha, p)
        # class_cost = self.single_class_cross_entropy(
        #     class_logits, tensor_labels, labeled_mask)


        distance_cost = linear_distance_penalty(
                z_dim, means, variances, probs, dataset.classes_num)
        # l2_means_cost = 0.0 * tensor_distance_weight * tf.reduce_mean(tf.sqrt(norm_squared(means)))
        # distance_cost = distance_cost + l2_means_cost

        # unsupervised_cost = tf.reduce_mean(reconstruction_error + log_dkl_cost)
        unsupervised_cost = tf.reduce_mean(
                rec_cost
                + log_cw_cost
                + tensor_distance_weight * distance_cost)

        full_cost = tf.reduce_mean(
                rec_cost
                + log_cw_cost
                + tensor_supervised_weight * class_cost
                + tensor_distance_weight * distance_cost)

        full_cec_cost = tf.reduce_mean(
                rec_cost
                + tensor_distance_weight * distance_cost
                + tensor_supervised_weight * cec_cost)

        tvars = tf.trainable_variables()
        freeze_vars = [
                var for var in tvars if
                "gmm_means" not in var.name
                and "gmm_betas" not in var.name]
        means_vars = [
                var for var in tvars if
                "gmm_means" in var.name
                or "gmm_betas" in var.name]

        # Prepare various train ops
        train_op = optimizer.minimize(full_cost)
        freeze_train_op = optimizer.minimize(full_cost, var_list=freeze_vars)
        means_train_op = optimizer.minimize(class_cost, var_list=means_vars)

        rec_train_op = optimizer.minimize(rec_cost)
        rec_dkl_train_op = optimizer.minimize(rec_cost + log_cw_cost)
        supervised_train_op = optimizer.minimize(
            class_cost + distance_cost * tensor_distance_weight)

        full_cec_train_op = optimizer.minimize(full_cec_cost)

        # Prepare variables for outside use
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.placeholders = {
            "X": tensor_x,
            "y": tensor_labels,
            "X_target": tensor_x_target,
            "train_labeled": train_labeled,
            "distance_weight": tensor_distance_weight,
            "supervised_weight": tensor_supervised_weight,
            "training": tensor_training,
            "gamma": tensor_gamma}

        self.out = {
            "logits": class_logits,
            "probs": class_probs,
            "z": tensor_z,
            "y": tensor_y}

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
            "unsupervised": unsupervised_cost}

        self.train_ops = {
            "full": train_op,
            "full_freeze": freeze_train_op,
            "supervised": supervised_train_op,
            "rec": rec_train_op,
            "means_only": means_train_op,
            "rec_dkl": rec_dkl_train_op,
            "full_cec": full_cec_train_op}

        self.train_op = train_op
        self.freeze_train_op = freeze_train_op
        self.supervised_train_op = supervised_train_op
        self.preds = class_logits


    def get_labels_mask(self, tensor_labels):
        one = tf.constant(1, tf.float32)
        labels_mask = tf.equal(one, tf.reduce_sum(tensor_labels, axis=-1))
        labels_mask = tf.cast(labels_mask, tf.float32)
        return labels_mask


    def ceclike_class_cost(self, tensor_z_all, tensor_target, labeled_mask,
                           class_logits, means, variances, probs, dataset):

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
        labeled_num = tf.reduce_sum(labeled_mask)

        classes_props = np.sum(tensor_target, 0)
        cwae_sum = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.eye(dataset.classes_num), logits=logits)
        cwae_sum *= classes_props / labeled_num

        return tf.reduce_sum(cwae_sum)

    def cwae_class_cost(self, tensor_z_all, tensor_target, labeled_mask,
                        means, variances, probs, dataset):

        D = tf.cast(tf.shape(tensor_z_all)[-1], tf.float32)
        G = tf.cast(tf.shape(means)[0], tf.float32)

        cwae_sum = tf.zeros([], dtype=tf.float32)
        tensor_z = tf.boolean_mask(tensor_z_all, tf.cast(labeled_mask, tf.bool))
        tensor_target = tf.boolean_mask(tensor_target, tf.cast(labeled_mask, tf.bool))
        argmaxed_tensor = tf.argmax(tensor_target, axis=-1)

        N0 = 100 # ???
        # gamma = tf.pow(4 / (3 * N0 / G), 0.4)
        gamma = self.gamma
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
        labeled_num = tf.reduce_sum(labeled_mask)

        cwae_sum = tf.cond(
            tf.greater(labeled_num, 0),
            lambda: tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.eye(dataset.classes_num), logits=logits)
                * classes_props / labeled_num,
            lambda: tf.zeros([], tf.float32))

        return tf.reduce_sum(cwae_sum)

# ======================================
# ========== HELPER FUNCTIONS ==========
# ======================================

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
        return(phi(s))
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


def get_gaussians(G, z_dim, init, dataset):
    with tf.variable_scope("gmm", reuse=tf.AUTO_REUSE):
        np.random.seed(25)
        means_initialization = tf.constant_initializer(
            np.random.normal(0, init, size=(G, z_dim)))
        np.random.seed()
        means = tf.get_variable(
                "gmm_means", [G, z_dim],
                initializer=means_initialization)
                # initializer=tf.random_uniform_initializer(-1.0, 1.0))

        # var_initialization = tf.constant_initializer(
        #     np.random.uniform(0.0, 1.0, size=(G)))
        # beta = tf.get_variable("gmm_betas", [G], initializer=var_initialization)
        beta = tf.zeros([G], name="betas")
        variances = 1.0 + tf.abs(beta)

        # Three ways to calculate p_k

        # labels_proportions = (
        #     dataset.labeled_train["y"].sum(0) / dataset.labeled_train["y"].sum())
        # probs = tf.constant(labels_proportions, dtype=tf.float32)
        probs = tf.constant([1 / G] * G, dtype=tf.float32)

        # logits = tf.get_variable("logit_probs", [G])
        # probs = tf.nn.softmax(logits)

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

def calculate_rev_logits(tensor_z, means, alpha, p):
    m_X_sub_matrix = tf.subtract(tf.expand_dims(means, 0), tf.expand_dims(tensor_z, 1))
    C1 = norm_squared(m_X_sub_matrix, axis=2)
    C2 = 1 / (C1 / (2 * alpha) + 0.01)
    return C2

def calculate_cw_logits(tensor_z, means, alpha, p, gamma):
    N = tf.cast(tf.shape(tensor_z)[0], tf.float32)
    D = tf.cast(tf.shape(tensor_z)[1], tf.float32)
    N0 = 100
    G = 10

    m_X_sub_matrix = tf.subtract(tf.expand_dims(means, 0), tf.expand_dims(tensor_z, 1))
    C1 = norm_squared(m_X_sub_matrix, axis=2)
    C2 = phi_d(C1 / (2 * (alpha + 2 * gamma)), D)
    C = 2 * p * 1 / G / tf.sqrt(2 * pi * (alpha + 2 * gamma)) * C2
    A = (1 / (G * G)) / tf.sqrt(4 * pi * gamma)
    B = tf.square(p) / tf.sqrt(2 * pi * (2 * alpha + 2 * gamma))

    C3 = C2 # + tf.log(p) / 100
    print(C3.shape)
    return -(A + B - C)
    # C = tf.reduce_sum(C3)

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
    class_logits = -norm_squared(tf.expand_dims(tensor_z, 1) - tf.expand_dims(means, 0), axis=-1) / (2 * alpha )
    # class_logits = tf.log(p) + -1/2 * tf.log(2 * pi * alpha) + class_logits
    class_logits = tf.nn.softmax(class_logits)
    return class_logits

def calculate_logits(tensor_z, means, alpha, p):
    class_logits = -norm_squared(tf.expand_dims(tensor_z, 1) - tf.expand_dims(means, 0), axis=-1) / (2 * alpha)
    class_logits = tf.log(p) - 0.5 * tf.log(2 * pi * alpha) + class_logits
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
        -tf.reduce_sum(tensor_target * tf.log(class_probs + 1e-5), reduction_indices=[1]))

    class_cost = tf.where(
        tf.less(class_cost, eps),
        tf.zeros_like(class_cost),
        class_cost)
    denominator = tf.reduce_sum(tensor_target)
    denominator = tf.cond(tf.equal(denominator, 0), lambda: tf.constant(1, tf.float32), lambda: denominator)
    # class_cost_fin = tf.reduce_mean(class_cost * non_zero_indices)
    class_cost_fin = tf.reduce_sum(class_cost * labeled_mask) / denominator
    return class_cost_fin

def calculate_logits_cost(
    class_logits, tensor_target,
    labeled_mask, eps=0):
    class_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=class_logits, labels=tensor_target)
    class_cost = tf.where(
        tf.less(class_cost, eps),
        tf.zeros_like(class_cost),
        class_cost)
    denominator = tf.reduce_sum(tensor_target)
    denominator = tf.cond(tf.equal(denominator, 0), lambda: tf.constant(1, tf.float32), lambda: denominator)
    # class_cost_fin = tf.reduce_mean(class_cost * non_zero_indices)
    class_cost_fin = tf.reduce_sum(class_cost * labeled_mask) / denominator
    return class_cost_fin

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
    return -tf.reduce_mean(dist) # sqrt

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
    G = 10
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
