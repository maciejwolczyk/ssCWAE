import matplotlib.pyplot as plt
import numpy as np
import torch

# from sklearn.metrics import adjusted_rand_score
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from torch.distributions.multivariate_normal import MultivariateNormal


def draw_gmm(model, loader, device, prefix=""):
    all_encoded = []
    all_labels = []
    for idx, (batch_X, batch_y) in enumerate(loader):
        batch_X = batch_X.to(device)
        encoded, _ = model(batch_X)
        all_encoded += [encoded.cpu().detach().numpy()]
        all_labels += [batch_y.numpy()]
        if idx > 10:
            break
    all_encoded = np.concatenate(all_encoded, 0)
    all_labels = np.concatenate(all_labels, 0)

    if all_encoded.shape[1] == 2:  # If we're on a plane, PCA is not necessary
        points = all_encoded
        means = model.gmm.means.cpu().detach().numpy()
    else:
        pca = PCA(2)
        points = pca.fit_transform(all_encoded)
        means = model.gmm.means.cpu().detach().numpy()
        means = pca.transform(means)

    handles = []
    if model.gmm.multilabel:
        colors = np.array(["cyan", "magenta", "yellow", "lime", "purple"])

        # labels = ["Not even, small", "Even, small", "Not even, big", "Even, big"]
        # for l, c in zip(labels, colors):
        #     handles += [Line2D([0], [0], marker='o', color=c, label=l)]

        new_labels = all_labels.copy()
        new_labels[:, 1] *= 2
        new_labels = new_labels.sum(1)

        plt.scatter(
            points[:, 0],
            points[:, 1],
            c=colors[new_labels]
        )

        # for task_idx in range(tasks_num):
        #     indices = all_labels[:, task_idx] == 0.
        #     class_encoded = points[indices]
        #     plt.scatter(
        #         class_encoded[:, 0],
        #         class_encoded[:, 1],
        #         c=colors[2 * task_idx]
        #     )
    else:
        plt.scatter(points[:, 0], points[:, 1])

    for comp_idx in range(len(means)):
        colors = ["r", "b", "g"]

        marker_type = "P" if comp_idx % 2 else "o"
        color = colors[comp_idx // 2]

        plt.scatter(
            means[comp_idx, 0], means[comp_idx, 1],
            marker=marker_type,
            c=color,
            s=50,
            edgecolors="k"
        )
    handles += [Line2D([0], [0], marker='P', color="r", label="Even")]
    handles += [Line2D([0], [0], marker='P', color="b", label="Big")]
    plt.legend(handles=handles)

    plt.legend(handles=handles)
    plt.savefig(f"results/{model.name}/{prefix}scatter.png")
    plt.close()


def show_reconstructions(model, loader, device, prefix=""):
    batch_X, _ = next(iter(loader))
    batch_X = batch_X.to(device)
    _, decoded = model(batch_X)
    decoded = decoded.detach().cpu()

    for idx, (rec, orig) in enumerate(zip(decoded[:10], batch_X[:10])):
        orig = orig.reshape(batch_X[0].shape).cpu().squeeze()
        rec = rec.reshape(batch_X[0].shape).cpu().squeeze()
        if len(orig.shape) > 2:
            orig = orig.permute(1, 2, 0)
            rec = rec.permute(1, 2, 0)

        plt.imshow(orig)
        plt.savefig(f"results/{model.name}/{prefix}rec_{idx}orig.png")
        plt.close()

        plt.imshow(rec)
        plt.savefig(f"results/{model.name}/{prefix}rec_{idx}rec.png")
        plt.close()


def sample_from_classes(model, num_per_class, device, prefix=""):
    samples = []
    for comp_idx in range(model.gmm.num_components):
        cov_matrix = torch.eye(model.gmm.latent_dim) * model.gmm.variances[comp_idx]
        cov_matrix = cov_matrix.to(device)
        dist = MultivariateNormal(model.gmm.means[comp_idx], cov_matrix)
        num_samples = torch.tensor([num_per_class]).to(device)
        samples += [dist.sample(num_samples)]

    stacked_samples = torch.cat(samples, 0).to(device)

    # TODO: pass image shape
    im_h, im_w, im_c = 64, 64, 3
    decoded = model.coder.decode(stacked_samples).view(-1, im_c, im_h, im_w)
    decoded = decoded.permute(0, 2, 3, 1)
    decoded = decoded.cpu().detach().numpy()
    canvas = create_image_matrix(decoded, row_len=num_per_class)
    plt.imshow(canvas)
    plt.savefig(f"results/{model.name}/{prefix}samples.png")
    plt.close()

    return decoded


def test_multiclass_sampling(model, device, prefix):
    num_components = model.gmm.num_components
    for first_idx in range(num_components - 2):
        start_rest_idx = (first_idx // 2) * 2 + 2
        for second_idx in range(start_rest_idx, num_components):
            sample_interpolation(
                model, first_idx, second_idx, device, prefix=prefix)
            sample_interpolation(
                model, first_idx, second_idx, device,
                interp_samples=True, prefix=prefix + "samp"
            )
            sample_intersection(
                model, first_idx, second_idx, device,
                prefix
            )

def sample_intersection(model, first_idx, second_idx, device, prefix=""):
    eye_matrix = torch.eye(model.gmm.latent_dim).to(device)
    first_mean = model.gmm.means[first_idx]
    first_cov = eye_matrix * model.gmm.variances[first_idx]
    first_cov_inv = torch.inverse(first_cov)

    second_mean = model.gmm.means[second_idx]
    second_cov = eye_matrix * model.gmm.variances[second_idx]
    second_cov_inv = torch.inverse(second_cov)

    new_cov = torch.inverse(first_cov_inv + second_cov_inv)
    new_mean = (
        (new_cov @ first_cov_inv) @ first_mean
        + (new_cov @ second_cov_inv) @ second_mean
    )
    dist = MultivariateNormal(new_mean, new_cov)

    samples = dist.sample(torch.tensor([20]).to(device))  # (20, D)

    # TODO: pass image shape
    im_h, im_w, im_c = 64, 64, 3
    decoded = model.coder.decode(samples).view(-1, im_c, im_h, im_w)
    decoded = decoded.permute(0, 2, 3, 1)
    decoded = decoded.cpu().detach().numpy()
    canvas = create_image_matrix(decoded, row_len=10)

    plt.imshow(canvas)
    plt.savefig(f"results/{model.name}/{prefix}intersec{first_idx}_{second_idx}.png")
    plt.close()

    return decoded


def sample_interpolation(model, first_idx, second_idx, device, interp_samples=False, prefix=""):
    eye_matrix = torch.eye(model.gmm.latent_dim).to(device)
    first_mean = model.gmm.means[first_idx]
    first_cov = eye_matrix * model.gmm.variances[first_idx]
    first_gaussian = MultivariateNormal(first_mean, first_cov)

    second_mean = model.gmm.means[second_idx]
    second_cov = eye_matrix * model.gmm.variances[second_idx]
    second_gaussian = MultivariateNormal(second_mean, second_cov)

    # S - samples
    # I - interpolation steps
    # D - latent dim
    if interp_samples:
        first_samples = first_gaussian.sample([10])  # (S, D)
        second_samples = second_gaussian.sample([10])  # (S, D)
        diffs = second_samples - first_samples  # (S, D)

        linspace = torch.linspace(0, 1, steps=10).to(device)  # (I)
        shifts = diffs.unsqueeze(1) * linspace.unsqueeze(-1)  # (S, I, D)
        stacked_samples = first_samples.unsqueeze(1) + shifts  # (S, I, D)
        stacked_samples = stacked_samples.view(100, -1)

    else:
        mean_diff = second_mean - first_mean  # (D)
        sample = first_gaussian.sample(torch.tensor([10]).to(device)) # (S, D)
        linspace = torch.linspace(0, 1, steps=10).to(device)
        shifts = linspace.unsqueeze(1) * mean_diff.unsqueeze(0) # (I, D)
        stacked_samples = sample.unsqueeze(1) + shifts.unsqueeze(0)  # (S, I, D)
        stacked_samples = stacked_samples.view(100, -1)

    # TODO: pass image shape
    im_h, im_w, im_c = 64, 64, 3
    decoded = model.coder.decode(stacked_samples).view(-1, im_c, im_h, im_w)
    decoded = decoded.permute(0, 2, 3, 1)
    decoded = decoded.cpu().detach().numpy()
    canvas = create_image_matrix(decoded)

    plt.imshow(canvas)
    plt.savefig(f"results/{model.name}/{prefix}interp{first_idx}_{second_idx}.png")
    plt.close()

    return decoded


def create_image_matrix(images: np.ndarray, row_len=None):
    N, H, W, C = images.shape
    if row_len is None:
        matrix_width = int(np.ceil(np.sqrt(N)))
        matrix_height = int(np.ceil(np.sqrt(N)))
    else:
        matrix_width = row_len
        matrix_height = int(np.ceil((N / matrix_width)))
    canvas = np.zeros(
        (matrix_height * H, matrix_width * W, C)
    )
    for idx, img in enumerate(images):
        row_idx = idx // matrix_width
        col_idx = idx % matrix_width

        canvas[
            row_idx * H:(row_idx + 1) * H,
            col_idx * W:(col_idx + 1) * W,
        ] = img

    canvas = canvas.squeeze()
    return canvas


def evaluate_model(model, loader, device, batch_num=None):
    correct_n = 0
    all_n = 0
    losses = []
    for batch_idx, (batch_X, batch_y) in enumerate(loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        encoded, decoded = model(batch_X)
        _, rec_loss, log_cw_loss = model.unsupervised_loss(encoded, decoded, batch_X)
        super_loss, nonweighted_super_loss = model.supervised_loss(encoded, batch_y)

        losses += [[rec_loss.item(), log_cw_loss.item(), nonweighted_super_loss.item()]]
        preds = model.gmm.calculate_logits(encoded)
        correct_n += (preds.argmax(1) == batch_y).float().sum().item()
        all_n += batch_y.numel()
        if batch_num is not None and batch_idx >= batch_num:
            break
    acc = correct_n / all_n
    return acc, np.mean(losses, 0)


def save_model(model, epoch_n):
    # TODO: optimizer?
    torch.save(model, f"results/{model.name}/weights_{epoch_n:03d}.ckpt")
