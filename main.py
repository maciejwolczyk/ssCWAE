import numpy as np
import torch

from itertools import cycle
from torch import optim
from tqdm import tqdm, trange

import architecture
import cwae
import data_loader
import metrics
import utils


def train_model(config):
    train_set, test_set, loaders = data_loader.get_dataset_by_name(
            config["dataset_name"],
            config["batch_size"],
            config["labeled_num"],
            config["multilabel"],
            config["rng_seed"]
    )

    if config["arch"] == "cnn":
        coder = architecture.CelebaCoder(**config["architecture"])
    else:
        coder = architecture.FCCoder(**config["architecture"])

    timestamp = utils.get_timestamp()
    model_name = (
        "{}/{}/{}{}d_cw{}_sw{}_bs{}_lr{}_rng{}_t{}").format(
            config["dataset_name"], coder.__class__.__name__,
            config["exp_name"],
            config["architecture"]["latent_dim"],
            config["loss_weights"]["cw"], config["loss_weights"]["supervised"],
            config["batch_size"], config["learning_rate"],
            config["rng_seed"], timestamp
    )

    utils.prepare_directories(model_name, config)

    class_probs = utils.get_class_probs(
        loaders["supervised"], multilabel=config["multilabel"]
    )
    gmm = cwae.GaussianMixture(
        config["components_num"],
        config["architecture"]["latent_dim"],
        config["init_radius"],
        update_probs=config["update_probs"],
        probs=class_probs,
        multilabel=config["multilabel"],
        separate_cw=config["separate_cw"]
    )

    model = cwae.Segma(model_name, gmm, coder, config["loss_weights"])
    model.to(config["device"])

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    run_training(model, optimizer, loaders, config["device"])


def run_training(model, optimizer, loaders, device):
    n_epochs = 200
    logger = utils.Logger(f"results/{model.name}/logs.txt")

    costs = []
    for epoch_n in trange(n_epochs + 1):
        cost = run_epoch(epoch_n, model, optimizer, loaders, logger, device)
        costs += [cost]


def run_epoch(epoch_n, model, optimizer, loaders, logger, device):
    labeled_iter = cycle(loaders["supervised"])
    losses = []
    for unlabeled_X, _ in tqdm(loaders["unsupervised"], leave=False):

        labeled_X, labeled_Y = next(labeled_iter)
        unlabeled_X = unlabeled_X.to(device)
        labeled_X, labeled_Y = labeled_X.to(device), labeled_Y.to(device)
        full_X = torch.cat((unlabeled_X, labeled_X), 0)

        optimizer.zero_grad()

        encoded, decoded = model(full_X)
        unsuper_loss, rec_loss, cw_loss = model.unsupervised_loss(
                encoded[:len(unlabeled_X)],
                decoded[:len(unlabeled_X)],
                unlabeled_X
        )
        super_loss, nonweighted_super_loss = model.supervised_loss(
                encoded[len(unlabeled_X):],
                labeled_Y
        )
        losses += [[rec_loss.item(), cw_loss.item(), nonweighted_super_loss.item()]]

        full_loss = unsuper_loss + super_loss

        full_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
        optimizer.step()

    if epoch_n % 5 == 0:
        prefix = f"e{epoch_n:03d}_"
        metrics.draw_gmm(model, loaders["supervised"], device, prefix=prefix+"lab_")
        metrics.draw_gmm(model, loaders["unsupervised"], device, prefix=prefix+"unlab_")
        metrics.show_reconstructions(model, loaders["unsupervised"], device, prefix=prefix)
        metrics.sample_from_classes(model, 10, device, prefix=prefix)
        metrics.test_multiclass_sampling(model, device, prefix)

        logger.write(f"\tEpoch {epoch_n}")
        for loader_name in loaders.keys():
            acc, losses = metrics.evaluate_model(model, loaders[loader_name], device, batch_num=10)
            logger.write(f"{loader_name}\tAcc: {acc:.3f}\tLosses: {losses}")

        means = model.gmm.means
        distances = (means.unsqueeze(0) - means.unsqueeze(1)).pow(2).sum(-1)
        distances = distances.detach().cpu().numpy()
        logger.write(f"Dist matrix:\n{distances}")

        logit_probs = model.gmm.logit_probs
        probs = torch.softmax(logit_probs, -1).detach().cpu().numpy()
        logger.write(f"Probs: {probs}")

        if epoch_n % 100 == 0 and epoch_n != 0:
            metrics.save_model(model, epoch_n)


def grid_train():
    config_path = "configs/three_celeba.yaml"
    config = utils.load_config(config_path)
    print(config)

    train_model(config)

if __name__ == "__main__":
    grid_train()
