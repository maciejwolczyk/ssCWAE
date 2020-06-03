import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import torch

from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Subset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

class SelectIndices(object):
    def __init__(self, indices):
        self.indices = indices

    def __call__(self, labels):
        return labels[self.indices]

class CombineClasses(object):
    def __init__(self, indices):
        self.indices = indices

    def __call__(self, labels):
        final_label = 0
        for idx, class_idx in enumerate(self.indices):
            final_label += int(labels[class_idx]) * (2 ** idx)
        return final_label


class MultiDataset(Dataset):
    def __init__(self, data, targets, transform=None, labels=None):
        self.data = data
        self.targets = targets.long()
        self.transform = transform
        self.labels = labels

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]

    def __len__(self):
        return len(self.data)


def prepare_loaders(train_dataset, test_dataset, batch_size, labeled_num, rng_seed):
    unsupervised_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=32,
    )

    rng = np.random.RandomState(seed=rng_seed)

    supervised_dataset = get_balanced_subset(train_dataset, labeled_num, rng)

    supervised_loader = DataLoader(
        supervised_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=16
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=16
    )
    loaders = {
        "unsupervised": unsupervised_loader,
        "supervised": supervised_loader,
        "test": test_loader
    }
    return loaders


def get_balanced_subset(train_dataset, labeled_num, rng):
    # TODO: fix
    if isinstance(train_dataset, datasets.MNIST):
        targets = train_dataset.targets.clone()
    if isinstance(train_dataset, ConcatDataset):  # SVHN
        targets = np.concatenate(
                [train_dataset.datasets[0].labels, train_dataset.datasets[1].labels], 0
            )
        targets = torch.tensor(targets)
    if isinstance(train_dataset, (MultiDataset, datasets.CelebA)):
        chosen_indices = rng.choice(len(train_dataset), labeled_num)
        subset = Subset(train_dataset, chosen_indices)
        return subset

    _, classes_counts = torch.unique(targets, return_counts=True)
    classes_probs = classes_counts.float() / classes_counts.sum()
    classes_num = len(classes_counts)

    # Decide how many samples should be in each class
    examples_per_class = (labeled_num * classes_probs).int()
    leftovers_num = labeled_num - examples_per_class.sum().item()
    sampled = np.random.multinomial(leftovers_num, classes_probs.tolist())

    examples_per_class += torch.tensor(sampled)
    examples_per_class = examples_per_class.tolist()
    assert sum(examples_per_class) == labeled_num

    chosen_indices = []
    for class_idx in range(classes_num):
        class_indices = torch.where(class_idx == targets)[0].tolist()
        chosen_indices += rng.choice(class_indices, size=examples_per_class[class_idx]).tolist()

    subset = Subset(train_dataset, chosen_indices)

    return subset

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


# TODO: merge get_mnist and get_multilabel_mnist
def get_multilabel_mnist(extra=True):
    mnist_train = datasets.MNIST(
            "./data", train=True, download=True,
            transform=ToTensor()
    )

    even_indices = (mnist_train.targets % 2 == 0).float()
    big_indices = (mnist_train.targets >= 5).float()
    train_labels = torch.stack((even_indices, big_indices), -1)

    multimnist_train = MultiDataset(
        mnist_train.data / 255.,
        train_labels,
        labels=["even", "big"]
    )

    mnist_test = datasets.MNIST(
            "./data", train=False, download=True,
            transform=ToTensor()
    )

    even_indices = (mnist_test.targets % 2 == 0).float()
    big_indices = (mnist_test.targets >= 5).float()
    test_labels = torch.stack((even_indices, big_indices), -1)
    multimnist_test = MultiDataset(
        mnist_test.data / 255.,
        test_labels,
        labels=["even", "big"]
    )

    labels = list(str(i) for i in range(10))
    return multimnist_train, multimnist_test, labels, "mnist"


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


def get_svhn(multilabel=False):
    svhn_train = datasets.SVHN(
            "./data", split="train", download=True,
            transform=ToTensor()
    )
    svhn_extra = datasets.SVHN(
            "./data", split="extra", download=True,
            transform=ToTensor()
    )
    svhn_train = ConcatDataset([svhn_train, svhn_extra])
    svhn_test = datasets.SVHN(
            "./data", split="test", download=True,
            transform=ToTensor()
    )
    labels = list(str(i) for i in range(0, 10))
    return svhn_train, svhn_test, labels, "svhn"


def get_celeba(multilabel=False):
    dataset_path = "/mnt/remote/wmii_gmum_projects/datasets/vision/CelebA_zipped/"
    chosen_indices = [20, 31, 15]  # Male, Smiling, Eyeglasses
    chosen_indices = torch.tensor(chosen_indices)

    celeba_transforms = transforms.Compose([
        transforms.CenterCrop(120),
        transforms.Resize(64),
        ToTensor()
    ])

    if multilabel:
        target_transform = SelectIndices(chosen_indices)
    else:
        target_transform = CombineClasses(chosen_indices)

    celeba_train = datasets.CelebA(
        dataset_path,
        split="train",
        target_type="attr",
        transform=celeba_transforms,
        target_transform=target_transform,
        download=True
    )
    celeba_test = datasets.CelebA(
        dataset_path,
        split="test",
        target_type="attr",
        transform=celeba_transforms,
        target_transform=target_transform,
        download=True
    )
    return celeba_train, celeba_test, None, "celeba"


def get_dataset_by_name(name, batch_size, labeled_num, multilabel, rng_seed):
    dataset_getters = {
       "MNIST": get_mnist,
       "Fashion_MNIST": get_fashion_mnist,
       "SVHN": get_svhn,
       "MultiMNIST": get_multilabel_mnist,
       "CelebA": get_celeba
    }

    getter = dataset_getters[name]
    train, test, labels, name = getter(multilabel)
    loaders = prepare_loaders(train, test, batch_size, labeled_num, rng_seed)
    return train, test, loaders


def one_hot_vectorize(dataset, labels_n):
    onehot_labels = np.zeros(dataset.shape + (labels_n,))
    for idx, y in enumerate(dataset):
        onehot_labels[idx, y] = 1
    return onehot_labels
