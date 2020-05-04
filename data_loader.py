import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

def prepare_loaders(train_dataset, test_dataset):
    unsupervised_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True)


    supervised_dataset = Subset(train_dataset, range(100))
    supervised_loader = DataLoader(
        supervised_dataset, batch_size=128, shuffle=True)

    test_loader = DataLoader(
        test_dataset, batch_size=128)
    loaders = {
        "unsupervised": unsupervised_loader,
        "supervised": supervised_loader,
        "test": test_loader
    }
            
    return loaders

def get_mnist(extra=True):
    mnist_train = datasets.MNIST(
            "./data", train=True, download=True,
            transform=ToTensor()
    )
    mnist_test = datasets.MNIST(
            "./data", train=False, download=True,
            transform=ToTensor()
    )
    labels = list(str(i) for i in range(10))
    return mnist_train, mnist_test, labels, "mnist"


def get_fashion_mnist(extra=True):
    fmnist_train = datasets.FashionMNIST(
            "./data", train=True, download=True,
            transform=ToTensor()
    )
    fmnist_test = datasets.FashionMNIST(
            "./data", train=False, download=True,
            transform=ToTensor()
    )

    labels = [
        "tshirt", "trousers", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle\nboot"]
    return fmnist_train, fmnist_test, labels, "fashion_mnist"


def get_svhn(extra=True):
    svhn_train = datasets.SVHN(
            "./data", train=True, download=True,
            transform=ToTensor()
    )
    # TODO: svhn_extra
    svhn_test = datasets.SVHN(
            "./data", train=False, download=True,
            transform=ToTensor()
    )
    labels = list(str(i) for i in range(1, 10)) + ["0"]
    return dataset_train, dataset_test, labels, "svhn"

def get_dataset_by_name(name, rng_seed, extra=True):
    dataset_getters = {
       "MNIST": get_mnist,
       "Fashion_MNIST": get_fashion_mnist,
       "SVHN": get_svhn,
       # "celeba_smiles": get_celeba_smiles,
       # "celeba_glasses": get_celeba_glasses,
       # "celeba_multitag": get_celeba_multitag,
       # "celeba_singletag": get_celeba_singletag
    }

    getter = dataset_getters[name]
    train, test, labels, name = getter(extra=extra)
    loaders = prepare_loaders(train, test)
    return train, test, loaders

def one_hot_vectorize(dataset, labels_n):
    onehot_labels = np.zeros(dataset.shape + (labels_n,))
    for idx, y in enumerate(dataset):
        onehot_labels[idx, y] = 1
    return onehot_labels
