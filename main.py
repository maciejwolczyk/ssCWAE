import itertools
import gc
import numpy as np
import torch
import yaml

from torch import optim
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

import architecture
import cwae
import data_loader
import metrics
import utils


METRICS_ENABLED = False
def train_model(config):
    train_set, test_set, loaders = data_loader.get_dataset_by_name(
            config["dataset_name"], config["batch_size"], config["labeled_num"])

    if config["dataset_name"] == "celeba_multitag" or config["dataset_name"] == "celeba_singletag":
        coder = architecture.CelebaCoder(
            dataset, kernel_num=kernel_num, h_dim=h_dim)
    else:
        coder = architecture.FCCoder(**config["architecture"])

    timestamp = utils.get_timestamp()
    model_name = (
        "{}/{}/{}d_bs{}_lr{}_rng{}_t{}").format(
            config["dataset_name"], coder.__class__.__name__,
            config["architecture"]["latent_dim"], config["batch_size"],
            config["learning_rate"], config["rng_seed"],
            timestamp
    )

    utils.prepare_directories(model_name, config)

    gmm = cwae.GaussianMixture(
        10,
        config["architecture"]["latent_dim"],
        config["init_radius"]
    )

    model = cwae.Segma(model_name, gmm, coder, config["loss_weights"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    run_training(model, optimizer, loaders)

def run_training(model, optimizer, loaders):
    n_epochs = 100

    costs = []
    for epoch_n in trange(n_epochs + 1):
        cost = run_epoch(epoch_n, model, optimizer, loaders)
        costs += [cost]

    if METRICS_ENABLED:
        costs = np.array(costs)
        train_costs, valid_costs, test_costs = costs[:, 0], costs[:, 1], costs[:, 2]

        metrics.save_samples(sess, model, dataset, 10000)
        metrics.save_costs(model, train_costs, "train")
        metrics.save_costs(model, valid_costs, "valid")
        metrics.save_costs(model, test_costs, "test")


def run_epoch(epoch_n, model, optimizer, loaders):
    losses = []
    for unlabeled_X, _ in tqdm(loaders["unsupervised"], leave=False):
        labeled_X, labeled_Y = next(iter(loaders["supervised"]))
        optimizer.zero_grad()

        encoded, decoded = model(unlabeled_X)
        unsuper_loss, rec_loss, cw_loss = model.unsupervised_loss(encoded, decoded, unlabeled_X)

        encoded, _ = model(labeled_X)
        super_loss = model.supervised_loss(encoded, labeled_Y)
        losses += [[rec_loss.item(), cw_loss.item(), super_loss.item()]]

        full_loss = unsuper_loss + super_loss
        full_loss.backward()
        optimizer.step()

    print(model.gmm.means)
    print("Mean loss", np.mean(losses, axis=0))

    if epoch_n % 1 == 0:
        metrics.draw_gmm(model, loaders, epoch_n=epoch_n)
        metrics.show_reconstructions(model, loaders["unsupervised"], epoch_n=epoch_n)

    train_acc = metrics.evaluate_model(model, loaders["supervised"])
    test_acc = metrics.evaluate_model(model, loaders["test"])
    print(f"\tEpoch {epoch_n}\tTrain acc: {train_acc}\tTest acc: {test_acc}")

    if METRICS_ENABLED:
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
    config = utils.load_yaml(config_path)

    train_model(config)
    gc.collect()

if __name__ == "__main__":
    grid_train()
