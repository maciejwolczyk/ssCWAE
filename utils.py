import numpy as np
import os
import time
import torch
import yaml
from argparse import ArgumentParser
from shutil import rmtree


class Logger:
    def __init__(self, filename):
        self.file = open(filename, "a")

    def write(self, string):
        print(string)
        print(string, file=self.file, flush=True)


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
    return int(time.time())


def apply_bernoulli_noise(x):
    return np.random.binomial(1, p=x, size=x.shape)


def apply_uniform_noise(x):
    return x + np.random.uniform(0, 1, size=x.shape)


def get_multilabel_probs(loader):
    # TODO: pass number of classes
    counts = torch.zeros(3)
    num_all = 0
    for _, y in loader:
        counts += y.sum(0)
        num_all += y.shape[0]

    probs = []
    for count in counts:
        probs += [(num_all - count) / num_all, count / num_all]
    probs = torch.tensor(probs)
    print(probs)
    probs /= probs.sum()
    print(probs)
    return probs


def get_class_probs(loader, multilabel=False):
    if multilabel:
        return get_multilabel_probs(loader)

    full_counts = {}
    for _, y in loader:
        keys, counts = torch.unique(y, return_counts=True)
        for key, count in zip(keys, counts):
            if key in full_counts:
                full_counts[key] += count
            else:
                full_counts[key] = count

    print(full_counts)
    full_counts = np.array(list(val for _, val in sorted(full_counts)))
    probs = full_counts.float() / full_counts.sum()
    print(probs)
    return probs


def str2bool(v):
    if v == "True":
        return True
    elif v == "False":
        return False
    else:
        raise ValueError


def load_config(config_path):
    config = load_yaml(config_path)
    parser = ArgumentParser()
    yaml_to_argparse(config, parser)
    config = parser.parse_args()
    config = argparse_to_dict(vars(config))
    return config


def load_yaml(config_path):
    with open(config_path) as f:
        data_loaded = yaml.safe_load(f)
    return data_loaded


def yaml_to_argparse(yaml_dict, parser, prefix=""):
    for key, item in yaml_dict.items():
        if not isinstance(item, dict):
            if type(item) == bool:
                type_ = str2bool
            else:
                type_ = type(item)
            parser.add_argument(f"--{prefix}{key}", default=item, type=type_)
        else:
            yaml_to_argparse(item, parser, prefix=f"{prefix}{key}.")


def argparse_to_dict(args):
    new_dict = {}
    for key, item in args.items():
        if "." in key:
            prefix, name = key.split(".", 1)
            if prefix not in new_dict:
                new_dict[prefix] = {name: item}
            else:
                new_dict[prefix][name] = item
        else:
            new_dict[key] = item
    return new_dict
