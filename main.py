import numpy as np
import itertools
import sys
import os
import collections

from tqdm import trange

import tensorflow as tf
from shutil import rmtree

import architecture
import baselines
import cwae
import data_loader
import metrics


frugal_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

def prepare_directories(model_name):
    weights_dir = "weights/{}_{}".format("dataset", "mnist_wae")
    os.makedirs(weights_dir, exist_ok=True)

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

def get_batch(batch_idx, batch_size, dataset):
    unlabeled_batch = dataset.unlabeled_train["X"][batch_idx * batch_size:(batch_idx + 1) * batch_size]

    # labeled_indices = np.random.choice(
    #     len(dataset.labeled_train["X"]), size=100, replace=False)

    labeled_batch_size = min(batch_size, len(dataset.labeled_train["X"]))

    indices_start = (batch_idx * labeled_batch_size) % len(dataset.labeled_train["X"])
    indices_end = ((batch_idx + 1) * labeled_batch_size) % len(dataset.labeled_train["X"])

    if labeled_batch_size >= len(dataset.labeled_train["X"]):
        labeled_indices = list(range(len(dataset.labeled_train["X"])))
    elif indices_start > indices_end:
        labeled_indices = (list(range(indices_start, len(dataset.labeled_train["X"])))
                           + list(range(0, indices_end)))
    else:
        labeled_indices = list(range(indices_start, indices_end))

    X_labeled = dataset.labeled_train["X"][labeled_indices]
    y_labeled = dataset.labeled_train["y"][labeled_indices]
    # print(y_labeled.sum(0))

    empty_labels = np.zeros((unlabeled_batch.shape[0], dataset.classes_num))

    X_batch = np.vstack((unlabeled_batch, X_labeled))
    y_batch = np.vstack((empty_labels, y_labeled))
    return X_batch, y_batch

def get_links_batch(batch_idx, batch_size, dataset):
    unlabeled_batch = dataset.train["X"][batch_idx * batch_size:(batch_idx + 1) * batch_size]

    input_dim = unlabeled_batch.shape[-1]
    labeled_batch_size = batch_size
    links_num = len(dataset.must_link) + len(dataset.cannot_link)
    indices = np.random.choice(
        links_num, size=min(labeled_batch_size // 2, links_num),
        replace=False)

    must_link_batch = []
    cannot_link_batch = []
    for idx in indices:
        if idx >= len(dataset.must_link):
            idx -= len(dataset.must_link)
            cannot_link_batch += [dataset.cannot_link[idx]]
        else:
            must_link_batch += [dataset.must_link[idx]]

    must_link_batch = np.array(must_link_batch).reshape(-1, input_dim)
    cannot_link_batch = np.array(cannot_link_batch).reshape(-1, input_dim)

    batch = np.concatenate((unlabeled_batch, must_link_batch, cannot_link_batch), 0)
    empty_labels = np.zeros((batch_size + labeled_batch_size, dataset.classes_num))

    must_link_labels = np.zeros((batch_size + labeled_batch_size,))
    must_link_labels[batch_size:batch_size + len(must_link_batch)] = 1

    cannot_link_labels = np.zeros((batch_size + labeled_batch_size,))
    cannot_link_labels[batch_size + len(must_link_batch):] = 1

    return batch, empty_labels, must_link_labels, cannot_link_labels


def train_model(
        dataset_name, latent_dim=300, batch_size=100,
        labeled_examples_n=100, h_dim=400, kernel_size=4,
        kernel_num=25, distance_weight=1.0, cc_ep=0.0, supervised_weight = 1.0,
        rng_seed=11, init=1.0, gamma=1.0, erf_weight=1.0, erf_alpha=0.05, links_num=1000):


    dataset = data_loader.get_dataset_by_name(dataset_name, rng_seed=rng_seed)
    # dataset.whitening()
    # dataset.remove_labels_fraction(
    #         number_to_keep=labeled_examples_n,
    #         keep_labels_proportions=True, batch_size=100)

    dataset.load_links(links_num)

    coder = architecture.WideShaoCoder(
            dataset, h_dim=h_dim,
            kernel_size=3,
            kernel_num=kernel_num)

    model_name = (
        "{}/{}/{}d_cwdist_dw{}_kn{}_hd{}_bs{}_sw{}" +
        "_{}links_e30_nonorm_pdf_subsets").format(
            dataset.name, coder.__class__.__name__, latent_dim,
            distance_weight, kernel_num, h_dim,
            batch_size, supervised_weight, links_num)

    print(model_name)
    prepare_directories(model_name)

    model = cwae.CwaeModel(
            model_name, coder, dataset,
            latent_dim=latent_dim,
            supervised_weight=supervised_weight,
            distance_weight=distance_weight, eps=cc_ep,
            init=init, gamma=gamma,
            erf_weight=erf_weight, erf_alpha=erf_alpha)

    run_training(model, dataset, batch_size)


def run_training(model, dataset, batch_size):
    n_epochs = 100
    with tf.Session(config=frugal_config) as sess:
        sess.run(tf.global_variables_initializer())
        costs = []

        for epoch_n in trange(n_epochs + 1):
            distance = True
            cost = run_epoch(epoch_n, sess, model, dataset, batch_size, distance)
            costs += [cost]

        costs = np.array(costs)
        valid_costs, test_costs = costs[:, 0], costs[:, 1] #, costs[:, 2]

        # metrics.save_costs(model, train_costs, "train")
        metrics.save_costs(model, valid_costs, "valid")
        metrics.save_costs(model, test_costs, "test")

def run_epoch(epoch_n, sess, model, dataset, batch_size, gamma_std):
    batches_num = int(np.ceil(len(dataset.train["X"]) / batch_size))
    # dataset.unlabeled_train if not links

    for batch_idx in trange(batches_num, leave=False):
        X_batch, y_batch, must_link, cannot_link = get_links_batch(
                batch_idx, batch_size, dataset)
        if dataset.name == "mnist" and False:
            noisy_X_batch = apply_bernoulli_noise(X_batch)
        else:
            noisy_X_batch = X_batch


        feed_dict = feed_dict={
            model.placeholders["X"]: noisy_X_batch,
            model.placeholders["X_target"]: X_batch,
            # model.placeholders["y"]: y_batch,
            model.placeholders["must_link"]: must_link,
            model.placeholders["cannot_link"]: cannot_link,
            model.placeholders["train_labeled"]: True,
            model.placeholders["training"]: True}

        if batch_idx % 500 == 0:
            gamma_val = sess.run(model.gamma, feed_dict=feed_dict)
        feed_dict[model.placeholders["gamma"]] = gamma_val

        if batch_idx % 300:
            feed_dict[model.placeholders["train_labeled"]] = False

        if epoch_n < 30:
            feed_dict[model.placeholders["distance_weight"]] = 0

        # if epoch_n < 35:
        #     feed_dict[model.placeholders["distance_weight"]] = 0
        #     feed_dict[model.placeholders["supervised_weight"]] = 0
        #     sess.run(model.train_ops["full"], feed_dict=feed_dict)
        # elif epoch_n == 35:
        #     sess.run(model.train_ops["means_only"], feed_dict=feed_dict)
        # elif epoch_n < 60:
        #     feed_dict[model.placeholders["distance_weight"]] = 0
        #     sess.run(model.train_ops["full"], feed_dict=feed_dict)
        # else:
        #     sess.run(model.train_ops["full"], feed_dict=feed_dict)

        sess.run(model.train_ops["full_cec_erf"], feed_dict=feed_dict)

        # if epoch_n < 10:
        #     sess.run(model.train_ops["rec_dkl"], feed_dict=feed_dict)
        # elif epoch_n == 10 and batch_idx < 100:
        #     sess.run(model.train_ops["means_only"], feed_dict=feed_dict)
        # else:
        #     sess.run(model.train_ops["full"], feed_dict=feed_dict)


        # if batch_idx % 50:
        #     print("\n", np.sum(
        #         (np.expand_dims(means, 0) - np.expand_dims(means, 1)) ** 2, axis=-1)[0], dist, "\n")


    if epoch_n % 5 == 0:
        print("\tGamma", gamma_val)

    # train_metrics, _ = metrics.evaluate_model(
    #     sess, model, dataset.semi_labeled_train,
    #     epoch_n, dataset, filename_prefix="train",
    #     subset=3000, training_mode=True)
    valid_metrics, valid_var = metrics.evaluate_model(
        sess, model, dataset.train,
        epoch_n, dataset, filename_prefix="valid",
        subset=3000, class_in_sum=False)
    test_metrics, _ = metrics.evaluate_model(
        sess, model, dataset.test,
        epoch_n, dataset, filename_prefix="test",
        subset=None)


    if epoch_n % 25 == 0:
        save_path = model.saver.save(
                sess, "weights/dataset_mnist_wae/epoch=%d.ckpt" % (epoch_n))
        print("Model saved in path: {}".format(save_path))

    if epoch_n % 10 == 0 and epoch_n > 1:
        metrics.interpolation(sess, model, dataset, epoch_n)
        metrics.sample_from_classes(sess, model, dataset, epoch_n, valid_var)
        metrics.sample_from_classes(sess, model, dataset, epoch_n, valid_var=None)

    return valid_metrics, test_metrics

if __name__ == "__main__":
    latent_dims = [64, 128]
    distance_weights = [1.]
    supervised_weights = [5.0]
    kernel_nums = [32, 64, 128]
    batch_sizes = [200]
    hidden_dims = [256, 512]
    gammas = [1.0]
    inits = [0.01]
    rng_seeds = [20]
    erf_weights = [0.]
    alphas = [1e-6]
    links_nums = [100, 500, 1000]

    for hyperparams in itertools.product(
            latent_dims, kernel_nums, distance_weights,
            hidden_dims, batch_sizes, rng_seeds,
            supervised_weights, inits, gammas,
            erf_weights, links_nums):
        ld, kn, dw, hd, bs, rs, sw, init, gamma, erf, ln = hyperparams
        train_model("mnist", latent_dim=ld, h_dim=hd,
            distance_weight=dw, kernel_num=kn, cc_ep=0.,
            batch_size=bs, labeled_examples_n=100, rng_seed=rs,
            supervised_weight=sw, init=init, gamma=gamma,
            erf_weight=erf, links_num=ln)
