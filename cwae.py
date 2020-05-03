import numpy as np
import tensorflow as tf
import tensorflow.train as tft
from math import pi

class GaussianMixture:
    def __init__(self, num_components, radius, var=0.01, requires_grad=False):
        # TODO: change init
        means = []
        for idx in range(num_components):
            rads = (2 * pi / num_components) * idx
            means += [[np.sin(rads) * radius, np.cos(rads) * radius]]
        self.num_components = num_components
        self.requires_grad = requires_grad
        self.means = torch.tensor(means, requires_grad=requires_grad, device=DEVICE)
        self.variances = torch.tensor([var] * num_components, device=DEVICE)
        self.probs = torch.tensor([1 / num_components] * num_components, device=DEVICE)

    def __len__(self):
        return self.num_components

    def parameters(self):
        if self.requires_grad:
            # TODO: self.variances? self.probs?
            return [self.means]
        else:
            return []

    def cluster(self, X):
        dists = (X.unsqueeze(1) - self.means.unsqueeze(0)) ** 2
        dists = dists.sum(-1)
        return dists.argmin(1)

    # TODO: torchify
    def calculate_logits(self, X):
        D = tf.cast(tf.shape(means)[-1], tf.float32)
        diffs = tf.expand_dims(tensor_z, 1) - tf.expand_dims(means, 0)
        class_logits = -norm_squared(diffs, axis=-1) / (2 * variance)
        class_logits = tf.log(p) - 0.5 * D * tf.log(2 * pi * variance) + class_logits
        return class_logits

    def cw_distance(self, X):
        N, D = X.shape

        gamma = torch.tensor(np.power(4 / (3 * X.shape[0] / self.num_components), 0.4)).to(DEVICE)

        variance_matrix = torch.unsqueeze(self.variances, 0) + torch.unsqueeze(self.variances, 1)
        X_sub_matrix = torch.unsqueeze(X, 0) - torch.unsqueeze(X, 1)
        A1 = torch.sum(X_sub_matrix ** 2, dim=2)

        A1 = torch.sum(phi_d(A1 / (4 * gamma), D))
        A = 1/(N*N * torch.sqrt(2 * pi * 2 * gamma)) * A1

        m_sub_matrix = torch.unsqueeze(self.means, 0) - torch.unsqueeze(self.means, 1)
        p_mul_matrix = torch.matmul(
            torch.unsqueeze(self.probs, 1),
            torch.unsqueeze(self.probs, 0)
        )
        B1 = torch.sum(m_sub_matrix ** 2, dim=2)
        B2 = phi_d(B1 / (2 * variance_matrix + 4 * gamma), D)
        B3 = p_mul_matrix / torch.sqrt(2 * pi * (variance_matrix + 2 * gamma))
        B = torch.sum(B3 * B2)

        m_X_sub_matrix = torch.unsqueeze(self.means, 0) - torch.unsqueeze(X, 1)
        C1 = torch.sum(m_X_sub_matrix ** 2, axis=2)
        C2 = phi_d(C1 / (2 * (self.variances + 2 * gamma)), D)
        C3 = 2 * self.probs / (N * torch.sqrt(2 * pi * (self.variances + 2 * gamma))) * C2
        C = torch.sum(C3)

        return torch.log(torch.mean(A + B - C))

class Segma(nn.Module):
    def __init__(self, name, gmm, coder, loss_weights):
        self.name = name
        self.gmm = gmm
        self.coder = coder
        self.loss_weight = loss_weights

    def supervised_loss(encoded, y_batch):
        class_logits = self.gmm.calculate_logits(encoded)
        classification_loss = self.class_loss(class_logits, y_batch)
        return classification_loss

    def unsupervised_loss(self, encoded, decoded, X):
        # TODO: train_labeled

        # MSE
        rec_cost = norm_squared(decoded, X_batch, dim=-1).mean()
        log_cw_cost = self.gmm.cw_distance(encoded)

        # TODO: gradient clipping

        # TODO: weightning
        return rec_cost + log_cw_cost

    def forward(x):
        return self.coder(x)

        


# ======================================
# ========== HELPER FUNCTIONS ==========
# ======================================

def get_labels_mask(tensor_labels):
    one = tf.constant(1, tf.float32)
    labels_mask = tf.equal(one, tf.reduce_sum(tensor_labels, axis=-1))
    labels_mask = tf.cast(labels_mask, tf.bool)
    return labels_mask


def norm_squared(X, dim=-1):
    return (X ** 2, dim=dim).sum()

def phi_f(s):
    t = s / 7.5
    return torch.exp(-s / 2) * (
            1 + 3.5156229*t**2 + 3.0899424*t**4 + 1.2067492*t**6
            + 0.2659732*t**8 + 0.0360768*t**10 + 0.0045813*t**12)


def phi_g(s):
    t = s / 7.5
    return torch.sqrt(2 / s) * (
            0.39894228 + 0.01328592*t**(-1) + 0.00225319*t**(-2)
            - 0.00157565*t**(-3) + 0.0091628*t**(-4) - 0.02057706*t**(-5)
            + 0.02635537*t**(-6) - 0.01647633*t**(-7) + 0.00392377*t**(-8))

def phi(x):
    a = torch.tensor(7.5).to(DEVICE)
    return phi_f(torch.min(x, a)) - phi_f(a) + phi_g(torch.max(x, a))

def phi_d(s, D):
    if D == 2:
        return phi(s)
    else:
        return 1 / torch.sqrt(1 + (4 * s) / (2 * D - 3))


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
