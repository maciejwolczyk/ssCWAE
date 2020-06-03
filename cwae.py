import numpy as np
from math import pi
import torch
from torch import nn
from torch.nn.functional import cross_entropy


class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, classes_num):
        super(MultiCrossEntropyLoss, self).__init__()
        self.classes_num = classes_num

    def forward(self, input, target):
        input = input.view(-1, self.classes_num, 2)
        loss_val = cross_entropy(input, target, reduction="mean")
        return loss_val


class Segma(nn.Module):
    def __init__(self, name, gmm, coder, loss_weights):
        super(Segma, self).__init__()
        self.name = name
        self.gmm = gmm
        self.coder = coder
        self.loss_weights = loss_weights
        self.class_loss = nn.CrossEntropyLoss()

    def supervised_loss(self, encoded, labels):
        class_logits = self.gmm.calculate_logits(encoded)
        classification_loss = self.class_loss(class_logits, labels)
        weighted_class_loss = classification_loss * self.loss_weights["supervised"]

        return weighted_class_loss, classification_loss

    def unsupervised_loss(self, encoded, decoded, X):
        X = X.view(decoded.shape)
        rec_loss = norm_squared(decoded - X).mean()
        log_cw_loss = self.gmm.cw_distance(encoded)

        unsuper_loss = rec_loss + log_cw_loss * self.loss_weights["cw"]
        return unsuper_loss, rec_loss, log_cw_loss

    def classify(self, x):
        x = self.coder.preprocess(x)
        encoded = self.coder.encode(x)
        logits = self.gmm.calculate_logits(encoded)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def forward(self, x):
        return self.coder(x)


class GaussianMixture(nn.Module):
    def __init__(
            self, num_components, latent_dim, radius, var=1.,
            update_probs=False, probs=None, multilabel=False, separate_cw=False):
        super(GaussianMixture, self).__init__()

        self.num_components = num_components
        self.latent_dim = latent_dim
        self.multilabel = multilabel
        self.separate_cw = separate_cw

        means = self.get_means_init(num_components, latent_dim, radius)

        self.means = nn.Parameter(torch.tensor(means).float())
        self.variances = nn.Parameter(
                torch.tensor([var] * num_components), requires_grad=False
        )

        if probs is None and separate_cw:
            probs = torch.tensor([0.5] * num_components)
        elif probs is None:
            probs = torch.tensor([1 / num_components] * num_components)

        if update_probs and not separate_cw:
            self.logit_probs = nn.Parameter(torch.log(probs), requires_grad=True)
        elif update_probs and separate_cw:
            raise NotImplementedError
        else:
            self.logit_probs = nn.Parameter(torch.log(probs), requires_grad=False)

    def get_means_init(self, num_components, latent_dim, radius):
        if latent_dim >= num_components:
            one_hot = np.zeros([num_components, latent_dim])
            one_hot[np.arange(latent_dim) % num_components, np.arange(latent_dim)] = 1
            one_hot *= radius / latent_dim * num_components
            init = one_hot + np.random.normal(0, .001, size=one_hot.shape)
        else:
            init = (np.random.rand(num_components, latent_dim) - 0.5) * 2 * radius / latent_dim
        return init

    def __len__(self):
        return self.num_components

    def parameters(self):
        if self.requires_grad:
            return [self.means]
        else:
            return []

    def cluster(self, X):
        dists = (X.unsqueeze(1) - self.means.unsqueeze(0)) ** 2
        dists = dists.sum(-1)
        return dists.argmin(1)

    def calculate_logits(self, X):
        D = self.means.shape[-1]
        diffs = torch.unsqueeze(X, 1) - torch.unsqueeze(self.means, 0)
        class_logits = -norm_squared(diffs, dim=-1) / (2 * self.variances)
        class_logits = torch.log_softmax(self.logit_probs, -1) - 0.5 * D * torch.log(2 * pi * self.variances) + class_logits
        if self.multilabel:
            class_logits = class_logits.view(-1, self.num_components // 2, 2)
            class_logits = class_logits.permute(0, 2, 1)
        return class_logits

    def cw_distance(self, X):
        if self.multilabel and self.separate_cw:
            cw_losses = []
            for k in range(self.num_components // 2):
                sub_means = self.means[2 * k:2 * k + 2]
                sub_vars = self.variances[2 * k:2 * k + 2]
                sub_probs = self.logit_probs[2 * k:2 * k + 2]
                sub_probs = torch.softmax(sub_probs, -1)

                cw_losses += [calculate_cw_distance(X, sub_means, sub_vars, sub_probs)]
            return sum(cw_losses) / len(cw_losses)

        else:
            probs = torch.softmax(self.logit_probs, -1)
            return calculate_cw_distance(X, self.means, self.variances, probs)


def calculate_cw_distance(X, means, variances, probs):
    N, D = X.shape
    K = len(means)

    gamma = np.power(4 / (3 * X.shape[0] / K), 0.4)

    variance_matrix = torch.unsqueeze(variances, 0) + torch.unsqueeze(variances, 1)
    X_sub_matrix = torch.unsqueeze(X, 0) - torch.unsqueeze(X, 1)
    A1 = torch.sum(X_sub_matrix ** 2, dim=2)

    A1 = torch.sum(phi_d(A1 / (4 * gamma), D))
    A = 1/(N*N * np.sqrt(2 * pi * 2 * gamma)) * A1

    m_sub_matrix = torch.unsqueeze(means, 0) - torch.unsqueeze(means, 1)
    p_mul_matrix = torch.matmul(
        torch.unsqueeze(probs, 1),
        torch.unsqueeze(probs, 0)
    )
    B1 = torch.sum(m_sub_matrix ** 2, dim=2)
    B2 = phi_d(B1 / (2 * variance_matrix + 4 * gamma), D)
    B3 = p_mul_matrix / torch.sqrt(2 * pi * (variance_matrix + 2 * gamma))
    B = torch.sum(B3 * B2)

    m_X_sub_matrix = torch.unsqueeze(means, 0) - torch.unsqueeze(X, 1)
    C1 = torch.sum(m_X_sub_matrix ** 2, axis=2)
    C2 = phi_d(C1 / (2 * (variances + 2 * gamma)), D)
    C3 = 2 * probs / (N * torch.sqrt(2 * pi * (variances + 2 * gamma))) * C2
    C = torch.sum(C3)

    return torch.log(torch.mean(A + B - C))



# ======================================
# ========== HELPER FUNCTIONS ==========
# ======================================

def norm_squared(X, dim=-1):
    return (X ** 2).sum(dim=dim)

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
    a = torch.tensor(7.5).to(x.device).float()
    return phi_f(torch.min(x, a)) - phi_f(a) + phi_g(torch.max(x, a))

def phi_d(s, D):
    if D == 2:
        return phi(s)
    else:
        return 1 / torch.sqrt(1 + (4 * s) / (2 * D - 3))
