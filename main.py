import itertools
import gc
import numpy as np
import torch
import yaml

from tqdm import trange

import architecture
import cwae
import data_loader
import metrics
import utils


def train_model(
        dataset_name, latent_dim=32, batch_size=100,
        labeled_examples_n=100, h_dim=400, kernel_size=4,
        kernel_num=25, distance_weight=1.0, cw_weight=1.0,
        supervised_weight=1.0, rng_seed=11, init=1.0,
        learning_rate=1e-3):

    dataset = data_loader.get_dataset_by_name(dataset_name, rng_seed=rng_seed)

    if dataset_name == "celeba_multitag" or dataset_name == "celeba_singletag":
        coder = architecture.CelebaCoder(
            dataset, kernel_num=kernel_num, h_dim=h_dim)
    else:
        coder = architecture.FCCoder(
            dataset, h_dim=h_dim, layers_num=kernel_num)

    model_name = (
        "{}/{}/{}d_kn{}_hd{}_bs{}_sw{}_dw{}_init{}" +
        "cw{}_lr{}_noneqprob_cyclic_realmse_4rep_rng{}_newlog").format(
            dataset.name, coder.__class__.__name__,
            latent_dim, kernel_num, h_dim,
            batch_size, supervised_weight, distance_weight,
            init, cw_weight, learning_rate, rng_seed)

    utils.prepare_directories(model_name)

    model = cwae.GmmCwaeModel(
            model_name, coder, dataset,
            z_dim=latent_dim, learning_rate=learning_rate,
            supervised_weight=supervised_weight,
            distance_weight=distance_weight, cw_weight=cw_weight,
            init=init)
    run_training(model, dataset, batch_size)


def run_training(model, dataset, batch_size):
    n_epochs = 400

    costs = []
    for epoch_n in trange(n_epochs + 1):
        cost = run_epoch(epoch_n,  model, dataset, batch_size)
        costs += [cost]

    costs = np.array(costs)
    train_costs, valid_costs, test_costs = costs[:, 0], costs[:, 1], costs[:, 2]

    metrics.save_samples(sess, model, dataset, 10000)
    metrics.save_costs(model, train_costs, "train")
    metrics.save_costs(model, valid_costs, "valid")
    metrics.save_costs(model, test_costs, "test")


def run_epoch(epoch_n, sess, model, dataset, batch_size):
    batches_num = len(dataset.unlabeled_train["X"]) // batch_size

    for unlabeled_X, unlabeled_Y in tqdm(train_loader, leave=False):
        labeled_X, labeled_Y = next(supervised_loader)
        optimizer.zero_grad()

        encoded, decoded = model(X_batch)
        unsuper_loss = model.unsupervised_loss(encoded, decoded, X_batch)
        super_loss = model.supervised_loss(encoded, y_batch)

        full_loss = unsuper_loss + super_loss
        full_loss.backward()
        optimizer.step()

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

    if epoch_n % 500 == 0 and epoch_n != 0:
        save_path = model.saver.save(
                sess, "results/{}/epoch={}.ckpt".format(model.name, epoch_n))
        print("Model saved in path: {}".format(save_path))

    if epoch_n % 10 == 0:
        if type(model).__name__ != "DeepGmmModel":
            analytic_mean = sess.run(model.gausses["means"])
            squared_diff = np.square(analytic_mean - valid_mean)
            mean_diff = np.sqrt(np.sum(squared_diff, axis=1))
            # print("Mean diff:", mean_diff)

        input_indices = list(range(10))
        metrics.interpolation(input_indices, sess, model, dataset, epoch_n)
        metrics.sample_from_classes(sess, model, dataset, epoch_n, valid_var=None)

        if type(model).__name__ != "CwaeModel":
            metrics.inter_class_interpolation(sess, model, dataset, epoch_n)
            metrics.cyclic_interpolation(
                    input_indices, sess, model, dataset, epoch_n)
            metrics.cyclic_interpolation(
                    input_indices, sess, model, dataset, epoch_n, direct=True)

    if epoch_n % 5 == 0:
        metrics.save_distance_matrix(sess, model, epoch_n)

    return train_metrics, valid_metrics, test_metrics


def grid_train():
    # TODO: yamlify
    # if dataset_name == "celeba_multitag" or dataset_name == "celeba_singletag":
    #     latent_dims = [32]
    #     learning_rates = [3e-4]
    #     cw_weights = [5.]
    #     distance_weights = [0.]
    #     supervised_weights = [10.]
    #     kernel_nums = [32]
    #     batch_sizes = [256]
    #     hidden_dims = [786]

    #     inits = [1.]
    #     rng_seeds = [20]

    config_path = "configs/mnist.yaml"
    config = yaml.load(config_path)
    print(config)
    die()


    for hyperparams in itertools.product(
            latent_dims, kernel_nums, distance_weights,
            hidden_dims, batch_sizes, cw_weights,
            rng_seeds, supervised_weights, inits, learning_rates):
        ld, kn, dw, hd, bs, cw, rs, sw, init, lr = hyperparams
        train_model(
            dataset_name, latent_dim=ld, h_dim=hd,
            distance_weight=dw, kernel_num=kn, cw_weight=cw,
            batch_size=bs, labeled_examples_n=labeled_num, rng_seed=rs,
            supervised_weight=sw, init=init, learning_rate=lr)
        gc.collect()

if __name__ == "__main__":
    grid_train()
