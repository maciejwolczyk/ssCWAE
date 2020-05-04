import numpy as np
import os
import yaml
import time
from shutil import rmtree

def prepare_directories(model_name, config):
    results_dir = "results/{}".format(model_name)
    if os.path.isdir(results_dir):
        rmtree(results_dir)
    os.makedirs(results_dir)

    graphs_dir = "results/{}/graphs".format(model_name)
    if os.path.isdir(graphs_dir):
        rmtree(graphs_dir)
    os.makedirs(graphs_dir)

    config_copy_path = "results/{}/config.yaml".format(model_name)
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f)

def get_timestamp():
    return time.time()

def load_yaml(config_path):
    with open(config_path) as f:
        data_loaded = yaml.safe_load(f)
    print(data_loaded)
    return data_loaded


def apply_bernoulli_noise(x):
    return np.random.binomial(1, p=x, size=x.shape)


def apply_uniform_noise(x):
    return x + np.random.uniform(0, 1, size=x.shape)
