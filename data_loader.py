import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")
import scipy.io as sio

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


class Dataset:
    def __init__(self, train_set, test_set, labels_names, dataset_name, rng_seed=11):
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


        self.classes_num = max(train_y.max(), test_y.max()) + 1

        self.train_examples_num = len(train_X)
        self.test_examples_num = len(test_X)

        self.image_shape = list(train_X.shape[1:])  # Ignore num of examples

        if len(self.image_shape) == 2:
            self.image_shape = self.image_shape + [1,]

        self.train = {"X": train_X.reshape(self.train_examples_num, -1),
                      "y": one_hot_vectorize(train_y, self.classes_num)}

        self.test = {"X": test_X.reshape(self.test_examples_num, -1),
                      "y": one_hot_vectorize(test_y, self.classes_num)}

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
            self.x_dim = len(self.train["X"][0])
            print("X DIM", self.x_dim)

        elif self.name == "svhn":
            reshaped_train = self.train["X"].reshape([-1] + self.image_shape)
            reshaped_test = self.test["X"].reshape([-1] + self.image_shape)

            print(reshaped_train.shape)
            self.std = reshaped_train.std(axis=(0, 1, 2))
            print("STD", self.std)
            reshaped_train /= self.std
            reshaped_test /= self.std

            self.train["X"] = reshaped_train.reshape(self.train["X"].shape)
            self.test["X"] = reshaped_test.reshape(self.test["X"].shape)

    def blackening(self, X):
        if self.name == "mnist":
            output = self.mean
            output[self.filtered] = X
        elif self.name == "svhn":
            reshaped_X = X.reshape([-1] + self.image_shape)
            reshaped_X *= self.std
            output = reshaped_X.reshape(X.shape)

        return output

    def load_links(self, pairs_num):
        pair_indices = self.rng.choice(
            self.train_examples_num, size=(10000, 2), replace=True)
        pair_indices = pair_indices[:pairs_num]
        print(pair_indices[0])

        links = []
        for first_idx, second_idx in pair_indices:
            X_pair = [self.train["X"][first_idx], self.train["X"][second_idx]]
            first_y = self.train["y"][first_idx].argmax()
            second_y = self.train["y"][second_idx].argmax()
            if first_y == second_y:
                plt.imshow(X_pair[0].reshape(28, 28))
                plt.imshow(X_pair[1].reshape(28, 28))
                links += [(X_pair, True)]
            else:
                links += [(X_pair, False)]
        self.links = links

        indices = pair_indices.reshape(-1)
        semi_labeled_X = self.train["X"][indices]
        semi_labeled_y = self.train["y"][indices]

        self.semi_labeled_train = {"X": semi_labeled_X,
                                   "y": semi_labeled_y}


    def remove_labels_fraction(
        self, number_to_keep=None,
        fraction_to_remove=0.9, keep_labels_proportions=True,
        batch_size=None):


        labels = np.copy(self.train["y"])
        labels_props = self.train["y"].sum(0) / self.train["y"].sum()
        
        # TODO: naprawić proporcje
        # labels_props = np.array([0.1] * 10)

        if keep_labels_proportions:
            argmax_labels = labels.argmax(-1).squeeze()
            labels_len = labels.shape[-1]

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
        remain_indices = np.where(remain_indices)[0]
        removed_indices = np.where(removed_indices)[0]
        self.rng.shuffle(remain_indices)
        self.rng.shuffle(removed_indices)
        print("Indices after shuffle", remain_indices, removed_indices)

        self.labeled_train = {"X": self.train["X"][remain_indices],
                              "y": self.train["y"][remain_indices]}
        self.unlabeled_train = {"X": self.train["X"][removed_indices]}
        print("Proportions in labeled sample",
            self.labeled_train["y"].sum(0) / self.labeled_train["y"].sum(),
            self.labeled_train["y"].sum())

        self.labeled_examples_num = len(self.labeled_train["X"])
        self.unlabeled_examples_num = len(self.unlabeled_train["X"])

        semi_labeled_X = np.vstack(
            (self.labeled_train["X"], self.unlabeled_train["X"]))
        dummy_y = np.zeros(
            (self.unlabeled_examples_num, self.classes_num))
        semi_labeled_y = np.vstack((self.labeled_train["y"], dummy_y))

        self.semi_labeled_train = {"X": semi_labeled_X,
                                   "y": semi_labeled_y}

    def reshuffle(self):
        np.random.shuffle(self.unlabeled_train["X"])



def get_mnist():
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    labels = list(str(i) for i in range(10))
    return mnist_train, mnist_test, labels, "mnist"


def get_fashion_mnist():
    fmnist_train, fmnist_test = tf.keras.datasets.fashion_mnist.load_data()

    onehot_labels = np.zeros(fmnist_train[1].shape + (10,))
    for idx, y in enumerate(fmnist_train[1]):
        onehot_labels[idx, y] = 1
    fmnist_train = np.reshape(fmnist_train[0] / 255, (-1, 28*28)), onehot_labels

    onehot_labels = np.zeros(fmnist_test[1].shape + (10,))
    for idx, y in enumerate(fmnist_test[1]):
        onehot_labels[idx, y] = 1
    fmnist_test = np.reshape(fmnist_test[0] / 255, (-1, 28*28)), onehot_labels
    labels = ["tshirt", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle\nboot"]
    return fmnist_train, fmnist_test, labels, "fashion_mnist"

def get_svhn(): # / 255?
    dataset_train = sio.loadmat("dataset/svhn/train_32x32.mat")
    dataset_train = dataset_train["X"].transpose(3, 0, 1, 2), dataset_train["y"] - 1
    dataset_test = sio.loadmat("dataset/svhn/test_32x32.mat")
    dataset_test = dataset_test["X"].transpose(3, 0, 1, 2), dataset_test["y"] - 1
    labels = list(str(i) for i in range(1, 10)) + [0]
    # print(np.histogram(dataset_train[1]))

    return dataset_train, dataset_test, labels, "svhn"

def get_norb(): # Doesn't work
    train_X = sio.loadmat("dataset/norb/smallnorb-train-dat.mat")
    train_y = sio.loadmat("dataset/norb/smallnorb-train-cat.mat")
    test_X = sio.loadmat("dataset/norb/smallnorb-test-dat.mat")
    test_y = sio.loadmat("dataset/norb/smallnorb-test-cat.mat")
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

def get_cifar():
    fmnist_train, fmnist_test = tf.keras.datasets.cifar10.load_data()

    onehot_labels = np.zeros((fmnist_train[1].shape[0], 10))
    for idx, y in enumerate(fmnist_train[1]):
        onehot_labels[idx, y] = 1
    fmnist_train = np.reshape(fmnist_train[0] / 255, (-1, 32*32*3)), onehot_labels

    onehot_labels = np.zeros((fmnist_test[1].shape[0], 10))
    for idx, y in enumerate(fmnist_test[1]):
        onehot_labels[idx, y] = 1

    fmnist_test = np.reshape(fmnist_test[0] / 255, (-1, 32*32*3)), onehot_labels
    labels = ["airplane", "automobile", "bird", "cat",
          "deer", "dog", "frog", "horse", "ship", "truck"]
    return fmnist_train, fmnist_test, labels, "fashion_mnist"

def get_celeba_images(examples_num):
    X = []
    for idx in tnrange(1, examples_num+1):
        a = plt.imread("dataset/img_align_celeba_64x64/{}.jpg".format(str(idx).zfill(6)))

        if len(a.shape) == 2:
            a = np.repeat(a, 3).reshape(64, 64, 3)
        X += [a / 255]
    # print(list(x.shape for x in X))
    return np.array(X).reshape(examples_num, -1)


def get_celeba_smiles():
    NUM_EXAMPLES = 199999

    X = get_celeba_images(NUM_EXAMPLES)

    Y = []
    with open("dataset/Smiling_CELEBA.tsv") as f:
        f.readline() # Omitting header
        for idx, line in enumerate(f):
            if idx >= NUM_EXAMPLES:
                break
            label = int(line.split("\t")[1])

            if label == 1:
                label = [0, 1]
            elif label == -1:
                label = [1, 0]
            else:
                raise ValueError("Ani jeden ani minus jeden: {}".format(label))
            Y += [label]
    Y = np.array(Y)
    print("Ratio of labels:", Y.sum(axis=0) / Y.shape[0])

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)
    return (train_x, train_y), (test_x, test_y), ["No smile", "Smile"], "celeba_smiles"


def get_celeba_glasses():
    NUM_EXAMPLES = 100000

    X = get_celeba_images(NUM_EXAMPLES)

    Y = []
    with open("dataset/Eyeglasses_CELEBA.tsv") as f:
        f.readline() # Omitting header
        for idx, line in enumerate(f):
            if idx >= NUM_EXAMPLES:
                break
            label = int(line.split("\t")[1])

            if label == 1:
                label = [0, 1]
            elif label == -1:
                label = [1, 0]
            else:
                raise ValueError("Ani jeden ani minus jeden: {}".format(label))
            Y += [label]
    Y = np.array(Y)
    print("Ratio of labels:", Y.sum(axis=0) / Y.shape[0])

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)
    return (train_x, train_y), (test_x, test_y), ["No glasses", "Glasses"], "celeba_glasses"


def get_dataset_by_name(name, rng_seed):
    dataset_getters = {
       "mnist": get_mnist,
       "fashion_mnist": get_fashion_mnist,
       "svhn": get_svhn,
       "cifar": get_cifar,
       "celeba_smiles": get_celeba_smiles,
       "celeba_glasses": get_celeba_glasses}

    getter = dataset_getters[name]
    train, test, labels, name = getter()
    dataset = Dataset(train, test, labels, name, rng_seed=rng_seed)
    return dataset

def one_hot_vectorize(dataset, labels_n):
    onehot_labels = np.zeros(dataset.shape + (labels_n,))
    for idx, y in enumerate(dataset):
        onehot_labels[idx, y] = 1
    return onehot_labels
