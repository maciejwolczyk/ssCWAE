import numpy as np
from math import pi
import torch
from torch import nn

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
        classification_loss *= self.loss_weights["supervised"]

        return classification_loss

    def unsupervised_loss(self, encoded, decoded, X):
        X = X.view(decoded.shape)
        rec_loss = norm_squared(decoded - X).mean()
        log_cw_loss = self.gmm.cw_distance(encoded)

        unsuper_loss = rec_loss + log_cw_loss * self.loss_weights["cw"]

        # TODO: gradient clipping
        return unsuper_loss, rec_loss, log_cw_loss

    def classify(self, x):
        x = self.coder.preprocess(x)
        encoded = self.coder.encoder(x)
        logits = self.gmm.calculate_logits(encoded)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def forward(self, x):
        return self.coder(x)

class GaussianMixture(nn.Module):
    def __init__(self, num_components, latent_dim, radius, var=1., probs=None):
        # TODO: change init
        super(GaussianMixture, self).__init__()

        self.num_components = num_components

        means = self.get_means_init(num_components, latent_dim, radius)

        self.means = nn.Parameter(torch.tensor(means).float())
        self.variances = nn.Parameter(torch.tensor([var] * num_components), requires_grad=False)

        if probs is None:
            probs = [1 / num_components] * num_components
        self.probs = nn.Parameter(torch.tensor(probs), requires_grad=False)

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
        class_logits = torch.log(self.probs) - 0.5 * D * torch.log(2 * pi * self.variances) + class_logits
        return class_logits

    def cw_distance(self, X):
        N, D = X.shape

        gamma = np.power(4 / (3 * 100 / self.num_components), 0.4)

        variance_matrix = torch.unsqueeze(self.variances, 0) + torch.unsqueeze(self.variances, 1)
        X_sub_matrix = torch.unsqueeze(X, 0) - torch.unsqueeze(X, 1)
        A1 = torch.sum(X_sub_matrix ** 2, dim=2)

        A1 = torch.sum(phi_d(A1 / (4 * gamma), D))
        A = 1/(N*N * np.sqrt(2 * pi * 2 * gamma)) * A1

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
