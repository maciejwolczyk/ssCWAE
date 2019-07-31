import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

plt.switch_backend("agg")


class Dataset:
    def __init__(
            self, train_set, test_set, labels_names,
            dataset_name, rng_seed=11):
        self.name = dataset_name
        self.labels_names = labels_names
        train_X, train_y = train_set
        test_X, test_y = test_set
        self.rng = np.random.RandomState(rng_seed)

        if train_X.max() > 1:
            train_X = train_X / 255
        if test_X.max() > 1:
            test_X = test_X / 255

        train_y = train_y.squeeze()
        test_y = test_y.squeeze()

        if len(train_y.shape) == 1:
            self.classes_num = max(train_y.max(), test_y.max()) + 1
            train_y = one_hot_vectorize(train_y, self.classes_num)
            test_y = one_hot_vectorize(test_y, self.classes_num)
        else:
            self.classes_num = train_y.shape[-1]

        self.train_examples_num = len(train_X)
        self.test_examples_num = len(test_X)

        self.image_shape = list(train_X.shape[1:])  # Ignore num of examples

        if len(self.image_shape) == 2:
            self.image_shape = self.image_shape + [1]

        self.train = {"X": train_X.reshape(self.train_examples_num, -1),
                      "y": train_y}

        self.test = {
                "X": test_X.reshape(self.test_examples_num, -1),
                "y": test_y
            }

        self.valid = {
                "X": self.train["X"][-5000:],
                "y": self.train["y"][-5000:]
            }

        print(self.train["y"].sum(0))
        self.whitened = False
        im_h, im_w, im_c = self.image_shape
        self.x_dim = im_h * im_w * im_c

    def whitening(self):
        if self.name == "mnist":
            self.whitened = True
            self.mean = self.train["X"].mean(axis=0)
            # self.std = 1
            std = self.train["X"].std(axis=0)
            self.filtered = np.where(std > 0.1)
            self.train["X"] = self.train["X"][:, std > 0.1]
            self.test["X"] = self.test["X"][:, std > 0.1]
            self.valid["X"] = self.valid["X"][:, std > 0.1]
            self.x_dim = len(self.train["X"][0])
            print("X DIM", self.x_dim)

        elif self.name == "svhn":
            reshaped_train = self.train["X"].reshape([-1] + self.image_shape)
            reshaped_test = self.test["X"].reshape([-1] + self.image_shape)

            reshaped_train += np.random.uniform(
                    0, 1. / 255., size=reshaped_train.shape)
            reshaped_test += np.random.uniform(
                    0, 1. / 255., size=reshaped_test.shape)

            print(reshaped_train.shape)
            self.std = reshaped_train.std(axis=(0, 1, 2))
            print("STD", self.std)
            reshaped_train /= self.std
            reshaped_test /= self.std

            self.train["X"] = reshaped_train.reshape(self.train["X"].shape)
            self.test["X"] = reshaped_test.reshape(self.test["X"].shape)
        else:
            raise NotImplementedError

    def blackening(self, X):
        if self.name == "mnist":
            output = self.mean
            output[self.filtered] = X
        elif self.name == "svhn":
            reshaped_X = X.reshape([-1] + self.image_shape)
            reshaped_X *= self.std
            output = reshaped_X.reshape(X.shape)

        return output


    def remove_labels_fraction(
            self, number_to_keep=None,
            fraction_to_remove=0.9, keep_labels_proportions=True,
            batch_size=None):

        labels = np.copy(self.train["y"])
        labels_props = self.train["y"].sum(0) / self.train["y"].sum()
        print(
            "Labels len:", len(labels),
            "Labels shape:", labels.shape,
            "Labels proportions:", labels_props
        )

        if keep_labels_proportions:
            argmax_labels = labels.argmax(-1).squeeze()

            class_to_keep = (labels_props * number_to_keep).astype("int")
            print("Before leftovers", class_to_keep, class_to_keep.sum())
            leftovers_n = number_to_keep - class_to_keep.sum()
            for idx in range(leftovers_n):
                diff = labels_props * 100 - class_to_keep
                class_to_keep[np.argmax(diff)] += 1
            print("After leftovers", class_to_keep, class_to_keep.sum())

            for idx, c_to_keep in enumerate(class_to_keep):
                class_examples = np.where(argmax_labels == idx)[0]

                if number_to_keep is None:
                    number_to_remove = int(len(class_examples) * (1 - fraction_to_remove))
                else:
                    number_to_remove = int(len(class_examples) - c_to_keep)
                indices = self.rng.choice(
                    class_examples, replace=False,
                    size=number_to_remove)

                # print(indices)
                labels[indices] = np.zeros(labels.shape[-1])
        else:
            examples_num = len(labels)
            if number_to_keep is None:
                number_to_remove = int(examples_num * (1 - fraction_to_remove))
            else:
                number_to_remove = examples_num - number_to_keep
            indices = self.rng.choice(
                examples_num, replace=False,
                size=number_to_remove)
            labels[indices] = np.zeros(labels.shape[-1])

        remain_indices = labels.sum(1).astype(bool)
        removed_indices = np.logical_not(remain_indices)
        print("Indices", remain_indices, removed_indices)
        print("Indices lens", len(remain_indices), len(removed_indices))

        if keep_labels_proportions:
            # Get remain indices per class
            argmaxed_y = self.train["y"].argmax(1)
            class_examples = []
            for val in range(self.classes_num):
                remain_val = np.logical_and(argmaxed_y == val, remain_indices)
                remain_val = np.where(remain_val)[0].tolist()
                class_examples += [remain_val]

            # Make every batch balanced
            batch_num = number_to_keep // batch_size
            remain_indices = []
            for batch_idx in range(batch_num + 1):
                for val in range(self.classes_num):
                    start_range = int(batch_idx * batch_size * labels_props[val])
                    end_range = int((batch_idx + 1) * batch_size * labels_props[val])
                    remain_indices += class_examples[val][start_range:end_range]
            remain_indices = np.array(remain_indices)

            print(self.train["y"][remain_indices].argmax(1))

        # Kolejnosc
        print("remain", remain_indices.shape, "removed", removed_indices.shape)
        remain_indices = np.where(remain_indices)[0]
        removed_indices = np.where(removed_indices)[0]
        self.rng.shuffle(remain_indices)
        self.rng.shuffle(removed_indices)
        print("Indices after shuffle", remain_indices, removed_indices)

        self.labeled_train = {"X": self.train["X"][remain_indices],
                              "y": self.train["y"][remain_indices]}

        self.unlabeled_train = {"X": self.train["X"].copy()}
        print(
            "Proportions in labeled sample",
            self.labeled_train["y"].sum(0) / self.labeled_train["y"].sum(),
            self.labeled_train["y"].sum())

        self.labeled_examples_num = len(self.labeled_train["X"])
        self.unlabeled_examples_num = len(self.unlabeled_train["X"])

        semi_labeled_X = np.vstack(
            (self.labeled_train["X"], self.train["X"][removed_indices]))
        dummy_y = np.zeros(
            (len(removed_indices), self.classes_num))
        semi_labeled_y = np.vstack((self.labeled_train["y"], dummy_y))

        self.semi_labeled_train = {"X": semi_labeled_X,
                                   "y": semi_labeled_y}

    def reshuffle(self):
        np.random.shuffle(self.unlabeled_train["X"])
        indices = np.random.permutation(len(self.labeled_train["X"]))
        self.labeled_train["X"] = self.labeled_train["X"][indices]
        self.labeled_train["y"] = self.labeled_train["y"][indices]


def get_mnist(extra=True):
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    labels = list(str(i) for i in range(10))
    return mnist_train, mnist_test, labels, "mnist"


def get_fashion_mnist(extra=True):
    fmnist_train, fmnist_test = tf.keras.datasets.fashion_mnist.load_data()

    onehot_labels = np.zeros(fmnist_train[1].shape + (10,))
    for idx, y in enumerate(fmnist_train[1]):
        onehot_labels[idx, y] = 1
    fmnist_train = np.reshape(fmnist_train[0] / 255, (-1, 28*28)), onehot_labels

    onehot_labels = np.zeros(fmnist_test[1].shape + (10,))
    for idx, y in enumerate(fmnist_test[1]):
        onehot_labels[idx, y] = 1
    fmnist_test = np.reshape(fmnist_test[0] / 255, (-1, 28*28)), onehot_labels
    labels = [
            "tshirt", "trousers", "pullover", "dress", "coat",
            "sandal", "shirt", "sneaker", "bag", "ankle\nboot"]
    return fmnist_train, fmnist_test, labels, "fashion_mnist"


def get_svhn(extra=True):
    dataset_dir = "dataset/svhn/"
    filenames = ["train_32x32.mat", "extra_32x32.mat", "test_32x32.mat"]
    if not all(os.path.isfile(dataset_dir + f) for f in filenames):
        raise ValueError(
                "No SVHN files in directory: {}. Files {} expected.".format(
                    dataset_dir, ", ".join(filenames)))
    dataset_train = sio.loadmat("dataset/svhn/train_32x32.mat")
    dataset_train = dataset_train["X"].transpose(3, 0, 1, 2), dataset_train["y"] - 1

    if extra:
        dataset_extra = sio.loadmat("dataset/svhn/extra_32x32.mat")
        dataset_extra = (
                dataset_extra["X"].transpose(3, 0, 1, 2),
                dataset_extra["y"] - 1
            )
        dataset_train = (
            np.concatenate([dataset_train[0], dataset_extra[0]], axis=0),
            np.concatenate([dataset_train[1], dataset_extra[1]], axis=0)
        )

    dataset_test = sio.loadmat("dataset/svhn/test_32x32.mat")
    dataset_test = (
            dataset_test["X"].transpose(3, 0, 1, 2),
            dataset_test["y"] - 1
        )
    labels = list(str(i) for i in range(1, 10)) + ["0"]
    # print(np.histogram(dataset_train[1]))

    return dataset_train, dataset_test, labels, "svhn"


def get_celeba_images(examples_num, extra=True):

    # TODO: poprawic to
    dataset_dir = "/mnt/users/mwolczyk/local/Repos/networks-do-networks/dataset/img_align_celeba/"
    orig_size = [178, 218]
    crop_size = [140, 140]
    target_size = [64, 64]

    start_y = (orig_size[1] - crop_size[0]) // 2
    start_x = (orig_size[0] - crop_size[1]) // 2

    train = []
    valid = []
    test = []

    images_list = sorted(os.listdir(dataset_dir))
    for idx, img_name in enumerate(tqdm(images_list)):
        if examples_num is not None and idx >= examples_num:
            break
        if not extra and idx > 20000 and idx < 182637:
            continue

        img = Image.open(dataset_dir + img_name).convert("RGB")
        img = img.crop((
            start_x,
            start_y,
            start_x + crop_size[0],
            start_y + crop_size[1]
        ))
        img = np.array(img.resize(target_size, Image.BILINEAR)) / 255
        if idx < 162770:
            train += [img]
        elif idx < 182637:
            valid += [img]
        else:
            test += [img]

    return np.array(train), np.array(valid), np.array(test)


# TODO: does not work as of yet
def get_celeba_multitag(extra=True):
    examples_num = 200000
    attr_labels = [
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive",
        "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
        "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
        "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
        "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open",
        "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
        "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns",
        "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
        "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
        "Young"
    ]
    dataset_dir = "/mnt/users/mwolczyk/local/Repos/networks-do-networks/dataset/"

    chosen_attributes = ["Heavy_Makeup", "Male", "Smiling"]
    chosen_indices = [idx for idx, label in enumerate(attr_labels)
                      if label in chosen_attributes]

    train_x, valid_x, test_x = get_celeba_images(examples_num, extra=extra)

    train_y = []
    valid_y = []
    test_y = []

    with open(dataset_dir + "/list_attr_celeba.txt") as f:
        f.readline()  # Omitting header
        f.readline()  # Omitting label list
        for idx, line in enumerate(f):
            if examples_num is not None and idx >= examples_num:
                break
            if not extra and idx > 20000 and idx < 182637:
                continue

            labels = line.split()[1:]  # skip filename in the first column
            one_hot_label = [0] * (len(chosen_attributes) + 1)

            for idx, attr_idx in enumerate(chosen_indices):
                val = int(labels[attr_idx])
                if val == 1:
                    one_hot_label[idx] = 1
                elif val == -1:
                    pass
                else:
                    raise ValueError("Neither 1 nor -1: {}".format(val))
            if idx < 162770:
                train_y += [one_hot_label]
            elif idx < 182637:
                valid_y += [one_hot_label]
            else:
                test_y += [one_hot_label]

    train_y = np.array(train_y)
    valid_y = np.array(valid_y)
    test_y = np.array(test_y)

    # If the example has no representation, pick
    Y[Y.sum(1) == 0, -1] = 1
    print("Ratio of labels:", Y.sum(axis=0) / Y.shape[0])
    print("Nonzero count", np.count_nonzero(Y.sum(1)))

    return (
        (train_x, train_y), (test_x, test_y),
        chosen_attributes + ["None"], "celeba_multitag")


def get_celeba_singletag(extra=True):
    examples_num = 200000
    attr_labels = [
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive",
        "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
        "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
        "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
        "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open",
        "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
        "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns",
        "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
        "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
        "Young"
    ]
    dataset_dir = "/mnt/users/mwolczyk/local/Repos/networks-do-networks/dataset/"

    chosen_attributes = ["Male", "Smiling"]
    chosen_indices = [idx for idx, label in enumerate(attr_labels)
                      if label in chosen_attributes]

    classes_num = 4
    labels_names = ["F/NS", "F/S", "M/NS", "M/S"]
    # labels_names = ["Not smiling", "Smiling"]

    train_y = []
    valid_y = []
    test_y = []
    with open(dataset_dir + "/list_attr_celeba.txt") as f:
        f.readline()  # Omitting header
        f.readline()  # Omitting label list
        for line_idx, line in enumerate(f):
            if examples_num is not None and line_idx >= examples_num:
                break
            if not extra and line_idx > 20000 and line_idx < 182637:
                continue

            labels = line.split()[1:]  # skip filename in the first column

            label_val = 0
            for idx, attr_idx in enumerate(chosen_indices):
                label_val *= 2
                val = int(labels[attr_idx])
                if val == 1:
                    label_val += 1
                elif val == -1:
                    pass
                else:
                    raise ValueError("Neither 1 nor -1: {}".format(label_val))

            one_hot_label = [0] * classes_num
            one_hot_label[label_val] = 1

            if line_idx < 162770:
                train_y += [one_hot_label]
            elif line_idx < 182637:
                valid_y += [one_hot_label]
            else:
                test_y += [one_hot_label]

    train_y = np.array(train_y)
    valid_y = np.array(valid_y)
    test_y = np.array(test_y)

    train_x, valid_x, test_x = get_celeba_images(examples_num, extra=extra)

    # If the example has no representation, pick
    # Y[Y.sum(1) == 0, -1] = 1
    # print("Ratio of labels:", Y.sum(axis=0) / Y.shape[0])
    # print("Nonzero count", np.count_nonzero(Y.sum(1)))

    return (train_x, train_y), (test_x, test_y), labels_names, "celeba_singletag"


def get_celeba_smiles(extra=True):
    if extra:
        num_examples = 20000
    else:
        num_examples = 199999

    X = get_celeba_images(num_examples)

    Y = []
    with open("dataset/Smiling_CELEBA.tsv") as f:
        f.readline()  # Omitting header
        for idx, line in enumerate(f):
            if idx >= num_examples:
                break
            label = int(line.split("\t")[1])

            if label == 1:
                label = [0, 1]
            elif label == -1:
                label = [1, 0]
            else:
                raise ValueError("Neither 1 nor -1: {}".format(label))
            Y += [label]
    Y = np.array(Y)
    print("Ratio of labels:", Y.sum(axis=0) / Y.shape[0])

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)
    labels_names = ["No smile", "Smile"]
    return (train_x, train_y), (test_x, test_y), labels_names, "celeba_smiles"


def get_celeba_glasses(extra=True):
    if extra:
        num_examples = 20000
    else:
        num_examples = 199999

    X = get_celeba_images(num_examples)

    Y = []
    with open("dataset/Eyeglasses_CELEBA.tsv") as f:
        f.readline()  # Omitting header
        for idx, line in enumerate(f):
            if idx >= num_examples:
                break
            label = int(line.split("\t")[1])

            if label == 1:
                label = [0, 1]
            elif label == -1:
                label = [1, 0]
            else:
                raise ValueError("Neither 1 nor -1: {}".format(label))
            Y += [label]
    Y = np.array(Y)
    print("Ratio of labels:", Y.sum(axis=0) / Y.shape[0])

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)
    labels_names = ["No glasses", "Glasses"]
    return (train_x, train_y), (test_x, test_y), labels_names, "celeba_glasses"


def get_dataset_by_name(name, rng_seed, extra=True):
    dataset_getters = {
       "mnist": get_mnist,
       "fashion_mnist": get_fashion_mnist,
       "svhn": get_svhn,
       "celeba_smiles": get_celeba_smiles,
       "celeba_glasses": get_celeba_glasses,
       "celeba_multitag": get_celeba_multitag,
       "celeba_singletag": get_celeba_singletag
    }

    getter = dataset_getters[name]
    train, test, labels, name = getter(extra=extra)
    dataset = Dataset(train, test, labels, name, rng_seed=rng_seed)
    return dataset


def one_hot_vectorize(dataset, labels_n):
    onehot_labels = np.zeros(dataset.shape + (labels_n,))
    for idx, y in enumerate(dataset):
        onehot_labels[idx, y] = 1
    return onehot_labels
