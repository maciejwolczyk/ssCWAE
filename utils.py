import numpy as np
import os
from shutil import rmtree


def prepare_directories(model_name):
    results_dir = "results/{}".format(model_name)
    if os.path.isdir(results_dir):
        rmtree(results_dir)
    os.makedirs(results_dir)

    graphs_dir = "results/{}/graphs".format(model_name)
    if os.path.isdir(graphs_dir):
        rmtree(graphs_dir)
    os.makedirs(graphs_dir)


def apply_bernoulli_noise(x):
    return np.random.binomial(1, p=x, size=x.shape)


def apply_uniform_noise(x):
    return x + np.random.uniform(0, 1, size=x.shape)
