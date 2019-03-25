import numpy as np
import itertools
import gc
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

def apply_uniform_noise(x):
     return x + np.random.uniform(0, 1, size=x.shape)

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
    # print(labeled_indices)

    X_labeled = dataset.labeled_train["X"][labeled_indices]
    y_labeled = dataset.labeled_train["y"][labeled_indices]
    # print(y_labeled.sum(0))

    empty_labels = np.zeros((unlabeled_batch.shape[0], dataset.classes_num))

    X_batch = np.vstack((unlabeled_batch, X_labeled))
    y_batch = np.vstack((empty_labels, y_labeled))
    # X_batch = X_labeled
    # y_batch = y_labeled
    return X_batch, y_batch

def train_model(
        dataset_name, latent_dim=300, batch_size=100,
        labeled_examples_n=100, h_dim=400, kernel_size=4,
        kernel_num=25, distance_weight=1.0, cc_ep=0.0, supervised_weight = 1.0,
        rng_seed=11, init=1.0, gamma=1.0, erf_weight=1.0,
        erf_alpha=0.05, labeled_super_weight=2.0, learning_rate=1e-3):


    dataset = data_loader.get_dataset_by_name(dataset_name, rng_seed=rng_seed)
    # dataset.whitening()
    dataset.remove_labels_fraction(
            number_to_keep=labeled_examples_n,
            keep_labels_proportions=True, batch_size=100)

    # coder = architecture.WideShaoClassifierCoder(
    #         dataset, h_dim=h_dim,
    #         kernel_size=3,
    #         kernel_num=kernel_num)

    coder = architecture.CifarCoder(
        dataset, kernel_num=kn, h_dim=h_dim)
    # coder = architecture.FCCoder(
    #     dataset, h_dim=h_dim, layers_num=kernel_num)
    classifier_cls = architecture.DummyClassifier

    model_name = (
        "{}/{}/{}/{}d_lindist_erfw{}_kn{}_hd{}_bs{}_sw{}_dw{}_a{}_gw{}_init{}" +
        "lr{}_realsig_nobn_clipping_recdiv100_onehotinit_samplingtest").format(
            dataset.name, coder.__class__.__name__, classifier_cls.__name__,
            latent_dim, erf_weight, kernel_num, h_dim,
            batch_size, supervised_weight, distance_weight, erf_alpha,
            gamma, init, lr)

    print(model_name)
    prepare_directories(model_name)

    model = cwae.CwaeModel(
            model_name, coder, dataset,
            latent_dim=latent_dim,
            supervised_weight=supervised_weight,
            distance_weight=distance_weight, eps=cc_ep,
            init=init, gamma_weight=gamma, classifier_cls=classifier_cls,
            erf_weight=erf_weight, erf_alpha=erf_alpha,
            labeled_super_weight=labeled_super_weight, learning_rate=lr)

    run_training(model, dataset, batch_size)


def run_training(model, dataset, batch_size):
    n_epochs = 150
    with tf.Session(config=frugal_config) as sess:
        sess.run(tf.global_variables_initializer())
        
        # TODO: restore session when using classifier
        # model.class_saver.restore(sess, "weights/classifier/epoch=100.ckpt")
        costs = []

        for epoch_n in trange(n_epochs + 1):
            distance = True
            cost = run_epoch(epoch_n, sess, model, dataset, batch_size, distance)
            costs += [cost]
            dataset.reshuffle()

        costs = np.array(costs)
        train_costs, valid_costs, test_costs = costs[:, 0], costs[:, 1], costs[:, 2]

        metrics.save_samples(sess, model, dataset, 10000)
        metrics.save_costs(model, train_costs, "train")
        metrics.save_costs(model, valid_costs, "valid")
        metrics.save_costs(model, test_costs, "test")

def run_epoch(epoch_n, sess, model, dataset, batch_size, gamma_std):
    # batches_num = int(np.ceil(len(dataset.train["X"]) / batch_size))
    batches_num = len(dataset.unlabeled_train["X"]) // batch_size
    # dataset.unlabeled_train if not links


    for batch_idx in trange(batches_num, leave=False):
        X_batch, y_batch = get_batch(batch_idx, batch_size, dataset)
        if dataset.name == "mnist":
            X_batch = apply_bernoulli_noise(X_batch)
        elif dataset.name == "svhn" and False:
            X_batch = apply_uniform_noise(X_batch)

        feed_dict = feed_dict={
            model.placeholders["X"]: X_batch,
            model.placeholders["y"]: y_batch,
            model.placeholders["training"]: True}

        if batch_idx % 300:
            feed_dict[model.placeholders["train_labeled"]] = False

        if epoch_n < 50:
            feed_dict[model.placeholders["erf_weight"]] = 0

        if epoch_n < 50:
            feed_dict[model.placeholders["classifier_distance_weight"]] = 0
            feed_dict[model.placeholders["distance_weight"]] = 0
        # elif epoch_n > 150:
        #     feed_dict[model.placeholders["classifier_distance_weight"]] = -1
        #     feed_dict[model.placeholders["distance_weight"]] = -1
            

        sess.run(model.train_ops["full"], feed_dict=feed_dict)

        # if epoch_n < 50:
        #     sess.run(model.train_ops["full_gmm_freeze"], feed_dict=feed_dict)
        # else:
        #     sess.run(model.train_ops["full_gmm"], feed_dict=feed_dict)

        # if epoch_n < 50:
        #     sess.run(model.train_ops["full_norm_freeze"], feed_dict=feed_dict)
        # else:
        #     sess.run(model.train_ops["full_norm"], feed_dict=feed_dict)

        # if batch_idx % 50:
        #     print("\n", np.sum(
        #         (np.expand_dims(means, 0) - np.expand_dims(means, 1)) ** 2, axis=-1)[0], dist, "\n")


    # TODO: przez training mode zawsze mamy accuracy 1.0
    train_metrics, _, _ = metrics.evaluate_model(
        sess, model, dataset.semi_labeled_train,
        epoch_n, dataset, filename_prefix="train",
        subset=3000, training_mode=False)
    valid_metrics, valid_var, valid_mean = metrics.evaluate_model(
        sess, model, dataset.valid,
        epoch_n, dataset, filename_prefix="valid",
        subset=3000, class_in_sum=False)
    test_metrics, _, _ = metrics.evaluate_model(
        sess, model, dataset.test,
        epoch_n, dataset, filename_prefix="test",
        subset=None)

    if epoch_n % 50 == 0:
        save_path = model.saver.save(
                sess, "weights/dataset_mnist_wae/epoch=%d.ckpt" % (epoch_n))
        print("Model saved in path: {}".format(save_path))

    if epoch_n % 5 == 0:
        analytic_mean = sess.run(model.gausses["means"])
        mean_diff = np.sqrt(np.sum(np.square(analytic_mean - valid_mean), axis=1))
        print("Mean diff:", mean_diff)

        metrics.interpolation(sess, model, dataset, epoch_n)
        metrics.sample_from_classes(sess, model, dataset, epoch_n, valid_var)
        metrics.sample_from_classes(sess, model, dataset, epoch_n, valid_var=None)

    return train_metrics, valid_metrics, test_metrics

if __name__ == "__main__":
    
    dataset_name = "cifar"

    if dataset_name == "mnist":
        labeled_num = 100
    elif dataset_name == "svhn":
        labeled_num = 1000
    elif dataset_name == "cifar":
        labeled_num = 4000
    
    # TODO: gradient clipping rzeczywiście?
    if dataset_name == "svhn" or dataset_name == "cifar":
        latent_dims = [128, 64]
        learning_rates = [5e-4]
        distance_weights = [0.]
        supervised_weights = [10., 100.]
        kernel_nums = [64]
        batch_sizes = [100]
        hidden_dims = [256]
        gammas = [1.]
        
        # init nie ma wiekszego znaczenia chyba
        inits = [1.]
        cc_eps = [0.0]
        # labeled_super_weights = [1.0]
        labeled_super_weights = [0.]
        rng_seeds = [20]
        erf_weights = [0.]
        alphas = [1e-3]

    elif dataset_name == "mnist":
        latent_dims = [5]
        distance_weights = [1., 1e2, 1e3, 1e4]
        supervised_weights = [1.0]
        kernel_nums = [4, 5, 6]
        # dla bs=200 tez spoko dziala
        batch_sizes = [100]
        hidden_dims = [100, 300, 500]
        gammas = [1.0]
        
        # init nie ma wiekszego znaczenia chyba
        inits = [0.1]
        cc_eps = [0.0]
        # labeled_super_weights = [1.0]
        labeled_super_weights = [0.]
        rng_seeds = [20]
        erf_weights = [0.]
        alphas = [1e-3]

    for hyperparams in itertools.product(
            latent_dims, kernel_nums, distance_weights,
            hidden_dims, batch_sizes, cc_eps, labeled_super_weights,
            rng_seeds, supervised_weights, inits, gammas,
            erf_weights, alphas, learning_rates):
        ld, kn, dw, hd, bs, ccep, lsw, rs, sw, init, gamma, erf, alpha, lr = hyperparams
        train_model(dataset_name, latent_dim=ld, h_dim=hd,
            distance_weight=dw, kernel_num=kn, cc_ep=ccep,
            batch_size=bs, labeled_examples_n=labeled_num, rng_seed=rs,
            supervised_weight=sw, init=init, gamma=gamma,
            erf_weight=erf, erf_alpha=alpha, labeled_super_weight=lsw, learning_rate=lr)
        gc.collect()
        # h = hpy()
        # print(h.heap())
