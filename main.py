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

    samples_dir = "results/{}/final_samples".format(model_name)
    os.makedirs(samples_dir)

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
    if dataset.name == "celeba_multitag":
        y_labeled = []
        for idx, label in enumerate(dataset.labeled_train["y"][labeled_indices]):
            positive_tags = (label == 1).nonzero()[0]
            tag = np.random.choice(positive_tags)
            one_hot_label = [0] * dataset.classes_num
            one_hot_label[tag] = 1
            y_labeled += [one_hot_label]
        y_labeled = np.array(y_labeled)
    else:
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
        kernel_num=25, distance_weight=1.0, cw_weight=1.0,
        supervised_weight=1.0, rng_seed=11, init=1.0, gamma=1.0,
        erf_weight=1.0, erf_alpha=0.05, labeled_super_weight=2.0,
        learning_rate=1e-3):


    dataset = data_loader.get_dataset_by_name(dataset_name, rng_seed=rng_seed)
    # dataset.whitening()

    keep_labels_proportions = False if dataset_name == "celeba_multitag" else True
    dataset.remove_labels_fraction(
            number_to_keep=labeled_examples_n,
            keep_labels_proportions=keep_labels_proportions,
            batch_size=100
    )

    # coder = architecture.WideShaoClassifierCoder(
    #         dataset, h_dim=h_dim,
    #         kernel_size=3,
    #         kernel_num=kernel_num)


    if dataset_name == "celeba_multitag" or dataset_name == "celeba_singletag":
        coder = architecture.CelebaCoder(
            dataset, kernel_num=kernel_num, h_dim=h_dim)
    else:
        coder = architecture.FCCoder(
            dataset, h_dim=h_dim, layers_num=kernel_num)

    classifier_cls = architecture.DummyClassifier
    # coder = architecture.FCClassifierCoder(
    #     dataset, h_dim=h_dim, layers_num=kernel_num)


    model_type = "gmmcwae"
    model_name = (
        "{}/{}/{}/{}d_erfw{}_kn{}_hd{}_bs{}_sw{}_dw{}_a{}_gw{}_init{}" +
        "cw{}_lr{}_noneqprob_cyclic_mse_2rep").format(
            dataset.name, coder.__class__.__name__, model_type,
            latent_dim, erf_weight, kernel_num, h_dim,
            batch_size, supervised_weight, distance_weight, erf_alpha,
            gamma, init, cw_weight, learning_rate)

    print(model_name)
    prepare_directories(model_name)

    # supervised_weight *= dataset.train_examples_num / labeled_examples_n
    print("Alpha waga!!!", supervised_weight, dataset.train_examples_num / labeled_examples_n)
    if model_type == "gmmcwae":
        model = cwae.GmmCwaeModel(
                model_name, coder, dataset,
                z_dim=latent_dim,
                supervised_weight=supervised_weight,
                distance_weight=distance_weight, cw_weight=cw_weight,
                init=init, gamma_weight=gamma, classifier_cls=classifier_cls,
                erf_weight=erf_weight, erf_alpha=erf_alpha,
                labeled_super_weight=labeled_super_weight, learning_rate=learning_rate)
    elif model_type == "onecwae":
        model = cwae.CwaeModel(
                model_name, coder, dataset, learning_rate=learning_rate,
                z_dim=latent_dim, gamma_weight=gamma,
                cw_weight=cw_weight)
    elif model_type == "multigmmcwae":
        model = cwae.MultiGmmCwaeModel(
                model_name, coder, dataset, learning_rate=learning_rate,
                z_dim=latent_dim, gamma_weight=gamma,
                cw_weight=cw_weight, supervised_weight=supervised_weight,
                init=init, gaussians_per_class=2)
    elif model_type == "deepgmm":
        model = cwae.DeepGmmModel(
                model_name, coder, dataset, m1=None,
                z_dim=latent_dim, supervised_weight=supervised_weight,
                learning_rate=learning_rate, cw_weight=cw_weight,
                init=init, gamma_weight=gamma)
    else:
        raise NotImplemented

    run_training(model, dataset, batch_size)


def run_training(model, dataset, batch_size):
    n_epochs = 500
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
        if dataset.name == "mnist" and dataset.whitened:
            X_batch = apply_bernoulli_noise(X_batch)
        elif dataset.name == "svhn" and False:
            X_batch = apply_uniform_noise(X_batch)

        if model.m1:
            X_batch = sess.run(model.m1.out["z"], {model.m1.placeholders["X"]: X_batch})

        feed_dict = feed_dict={
            model.placeholders["X"]: X_batch,
            model.placeholders["y"]: y_batch,
            model.placeholders["training"]: True}

        feed_dict[model.placeholders["train_labeled"]] = False

        if type(model).__name__ == "GmmCwaeModel":

            max_iters = batches_num * 150
            current_iter = batches_num * epoch_n + batch_idx
            progress = min(current_iter / max_iters, 1)

            current_cw_weight = progress * model.cw_weight
            # feed_dict[model.placeholders["cw_weight"]] = current_cw_weight

            if epoch_n < 50:
                feed_dict[model.placeholders["erf_weight"]] = 0
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


    if type(model).__name__ == "GmmCwaeModel":
        train_metrics, _, _ = metrics.evaluate_gmmcwae(
            sess, model, dataset.semi_labeled_train,
            epoch_n, dataset, filename_prefix="train",
            subset=3000, training_mode=False)
        valid_metrics, valid_var, valid_mean = metrics.evaluate_gmmcwae(
            sess, model, dataset.valid,
            epoch_n, dataset, filename_prefix="valid",
            subset=None, class_in_sum=False)
        test_metrics, _, _ = metrics.evaluate_gmmcwae(
            sess, model, dataset.test,
            epoch_n, dataset, filename_prefix="test",
            subset=None)
    elif type(model).__name__ == "CwaeModel":
        train_metrics, _, _ = metrics.evaluate_cwae(
            sess, model, dataset.semi_labeled_train,
            epoch_n, dataset, filename_prefix="train",
            subset=3000, training_mode=False)
        valid_metrics, valid_var, valid_mean = metrics.evaluate_cwae(
            sess, model, dataset.valid,
            epoch_n, dataset, filename_prefix="valid",
            subset=None, class_in_sum=False)
        test_metrics, _, _ = metrics.evaluate_cwae(
            sess, model, dataset.test,
            epoch_n, dataset, filename_prefix="test",
            subset=None)
    elif type(model).__name__ == "MultiGmmCwaeModel":
        train_metrics, _, _ = metrics.evaluate_multigmmcwae(
            sess, model, dataset.semi_labeled_train,
            epoch_n, dataset, filename_prefix="train",
            subset=3000, training_mode=False)
        valid_metrics, valid_var, valid_mean = metrics.evaluate_multigmmcwae(
            sess, model, dataset.valid,
            epoch_n, dataset, filename_prefix="valid",
            subset=None, class_in_sum=False)
        test_metrics, _, _ = metrics.evaluate_multigmmcwae(
            sess, model, dataset.test,
            epoch_n, dataset, filename_prefix="test",
            subset=None)
    elif type(model).__name__ == "DeepGmmModel":
        train_metrics, _, _ = metrics.evaluate_deepgmm(
            sess, model, dataset.semi_labeled_train,
            epoch_n, dataset, filename_prefix="train",
            subset=3000, training_mode=False)
        valid_metrics, valid_var, valid_mean = metrics.evaluate_deepgmm(
            sess, model, dataset.valid,
            epoch_n, dataset, filename_prefix="valid",
            subset=None, class_in_sum=False)
        test_metrics, _, _ = metrics.evaluate_deepgmm(
            sess, model, dataset.test,
            epoch_n, dataset, filename_prefix="test",
            subset=None)

    if epoch_n % 500 == 0:
        save_path = model.saver.save(
                sess, "results/{}/epoch={}.ckpt".format(model.name, epoch_n))
        print("Model saved in path: {}".format(save_path))

    if epoch_n % 10 == 0:
        print("Model name", type(model).__name__)
        if type(model).__name__ != "DeepGmmModel":
            analytic_mean = sess.run(model.gausses["means"])
            mean_diff = np.sqrt(np.sum(np.square(analytic_mean - valid_mean), axis=1))
            print("Mean diff:", mean_diff)

        metrics.interpolation(sess, model, dataset, epoch_n)
        metrics.sample_from_classes(sess, model, dataset, epoch_n, valid_var=None)

        if type(model).__name__ != "CwaeModel":
            metrics.inter_class_interpolation(sess, model, dataset, epoch_n)
            metrics.cyclic_interpolation(sess, model, dataset, epoch_n)
            metrics.cyclic_interpolation(sess, model, dataset, epoch_n, direct=True)
            # metrics.sample_from_classes(sess, model, dataset, epoch_n, valid_var)

    if epoch_n % 5 == 0:
        metrics.save_distance_matrix(sess, model, epoch_n)

    return train_metrics, valid_metrics, test_metrics


def load_and_test():
    # weights_filename = "20d_erfw0.0_kn5_hd786_bs1000_sw10.0_dw0.0_a0.001_gw1.0_init2.0cw5.0_lr0.0003_1000l_noalpha_noneqprob_nobn_cyclic_finaltest"
    # weights_filename = "20d_erfw0.0_kn5_hd786_bs1000_sw0.0_dw0.0_a0.001_gw1.0_init2.0cw5.0_lr0.0003_100l_noalpha_noneqprob_nobn_cyclic_nowhiten_rep_1000e_dataset"
    # weights_filename = "20d_erfw0.0_kn5_hd786_bs1000_sw5.0_dw0.0_a0.001_gw1.0_init2.0cw5.0_lr0.0003_100l_noalpha_noneqprob_nobn_cyclic_nowhiten_rep_1000e_dataset"
    weights_filename = "32d_erfw0.0_kn32_hd1024_bs256_sw10.0_dw0.0_a0.001_gw1.0_init1.0cw5.0_lr0.0003_100l_noalpha_noneqprob_nobn_cyclic_nowhiten_rep_1000e_dataset"

    model_type = "gmmcwae"
    model_name = "celeba_gmm_bigger_hd1024"
    epoch_n = 500
    dataset_name = "celeba_singletag"


    if dataset_name == "mnist":
        labeled_examples_num = 100
        latent_dim = 10
        supervised_weight = 1.
        kernel_num = 2
        learning_rate = 3e-4
        cw_weight = 5.
        batch_size = 100
        h_dim = 786
        init = 1.

    elif dataset_name == "svhn":
        labeled_examples_num = 1000
        latent_dim = 20
        learning_rate = 3e-4
        cw_weight = 5.
        supervised_weight = 10.
        kernel_num = 5
        batch_size = 1000
        h_dim = 786
        init = 2.

    elif dataset_name == "celeba_multitag" or dataset_name == "celeba_singletag":
        labeled_examples_num = 1000
        latent_dim = 32
        learning_rate = 3e-4
        cw_weight = 5.
        supervised_weight = 5.
        kernel_num = 32
        batch_size = 256
        h_dim = 1024
        init = 1.

    prepare_directories(model_name)
    dataset = data_loader.get_dataset_by_name(dataset_name, rng_seed=23, extra=False)
    dataset.remove_labels_fraction(
        number_to_keep=labeled_examples_num,
        keep_labels_proportions=True, batch_size=100)

    
    if dataset_name == "celeba_multitag" or dataset_name == "celeba_singletag":
        coder = architecture.CelebaCoder(
            dataset, kernel_num=kernel_num, h_dim=h_dim)
    else:
        coder = architecture.FCCoder(
            dataset, h_dim=h_dim, layers_num=kernel_num)

    classifier_cls = architecture.DummyClassifier

    if model_type == "onecwae":
        model = cwae.CwaeModel(
                model_name, coder, dataset, learning_rate=learning_rate,
                z_dim=latent_dim, gamma_weight=1.,
                cw_weight=cw_weight)
    else:
        model = cwae.GmmCwaeModel(
                model_name, coder, dataset,
                z_dim=latent_dim,
                supervised_weight=supervised_weight,
                distance_weight=0., cw_weight=cw_weight,
                init=init, gamma_weight=1., classifier_cls=classifier_cls,
                erf_weight=0., erf_alpha=0.01,
                labeled_super_weight=0., learning_rate=learning_rate)


    weights_path = "results/{}/{}/{}/{}/epoch={}.ckpt".format(
            dataset.name, coder.__class__.__name__, model_type, weights_filename, epoch_n) 
    
    with tf.Session(config=frugal_config) as sess:
        sess.run(tf.global_variables_initializer())
        model.saver.restore(sess, weights_path)
        metrics.interpolation(sess, model, dataset, "")
        # metrics.inter_class_interpolation(sess, model, dataset, "")
        metrics.sample_from_classes(sess, model, dataset, "", valid_var=None)
        metrics.save_samples(sess, model, dataset, 10000)

        if model_type != "onecwae":
            metrics.cyclic_interpolation(sess, model, dataset, "")
            metrics.cyclic_interpolation(sess, model, dataset, "", extrapolate=True)
            metrics.cyclic_interpolation(sess, model, dataset, "", direct=True)

def grid_train():
    dataset_name = "mnist"

    if dataset_name == "mnist":
        labeled_num = 100
    elif dataset_name == "svhn":
        labeled_num = 1000
        # labeled_num = 2000
        # labeled_num = 5000
        # labeled_num = 10000
    elif dataset_name == "cifar":
        labeled_num = 4000
    elif dataset_name == "celeba_multitag" or dataset_name == "celeba_singletag":
        labeled_num = 1000


    if dataset_name == "svhn":
        latent_dims = [20]
        learning_rates = [3e-4]
        distance_weights = [0.]
        cw_weights = [5.]
        supervised_weights = [0.]
        kernel_nums = [5]
        batch_sizes = [1000]
        hidden_dims = [786]
        gammas = [1.]

        inits = [2.]
        cc_eps = [0.0]
        # labeled_super_weights = [1.0]
        labeled_super_weights = [0.]
        rng_seeds = [20]
        erf_weights = [0.]
        alphas = [1e-3]

    elif dataset_name == "cifar":
        latent_dims = [16, 50]
        learning_rates = [5e-4]
        cw_weights = [5.]
        distance_weights = [0.]
        supervised_weights = [5.]
        kernel_nums = [64]
        batch_sizes = [128]
        hidden_dims = [512, 256]
        gammas = [1.]

        # im większy init tym gorszy FID
        inits = [10.]
        cc_eps = [0.0]
        # labeled_super_weights = [1.0]
        labeled_super_weights = [0.]
        rng_seeds = [20]
        erf_weights = [0.]
        alphas = [1e-3]

    elif dataset_name == "celeba_multitag" or dataset_name == "celeba_singletag":
        latent_dims = [32]
        learning_rates = [3e-4]
        cw_weights = [5.]
        distance_weights = [0.]
        supervised_weights = [10.]
        kernel_nums = [32]
        batch_sizes = [256]
        hidden_dims = [786]
        gammas = [1.]

        # im większy init tym gorszy FID
        inits = [1.]
        cc_eps = [0.0]
        # labeled_super_weights = [1.0]
        labeled_super_weights = [0.]
        rng_seeds = [20]
        erf_weights = [0.]
        alphas = [1e-3]

    elif dataset_name == "mnist":
        latent_dims = [10]
        distance_weights = [0.]
        supervised_weights = [0.]
        kernel_nums = [2]

        learning_rates = [3e-4]
        cw_weights = [5.]
        # dla bs=200 tez spoko dziala
        batch_sizes = [100]
        hidden_dims = [1024]
        gammas = [1.0]

        # init nie ma wiekszego znaczenia chyba
        inits = [0.1]
        cc_eps = [0.0]
        # labeled_super_weights = [1.0]
        labeled_super_weights = [0.]
        rng_seeds = [25]
        erf_weights = [0.]
        alphas = [1e-3]

    for hyperparams in itertools.product(
            latent_dims, kernel_nums, distance_weights,
            hidden_dims, batch_sizes, cw_weights, labeled_super_weights,
            rng_seeds, supervised_weights, inits, gammas,
            erf_weights, alphas, learning_rates):
        ld, kn, dw, hd, bs, cw, lsw, rs, sw, init, gamma, erf, alpha, lr = hyperparams
        train_model(dataset_name, latent_dim=ld, h_dim=hd,
            distance_weight=dw, kernel_num=kn, cw_weight=cw,
            batch_size=bs, labeled_examples_n=labeled_num, rng_seed=rs,
            supervised_weight=sw, init=init, gamma=gamma,
            erf_weight=erf, erf_alpha=alpha, labeled_super_weight=lsw, learning_rate=lr)
        gc.collect()
        # h = hpy()
        # print(h.heap())

if __name__ == "__main__":
    grid_train()

