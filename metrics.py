import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from main import get_links_batch

def draw_gmm(
        z, y, dataset, means=None, alpha=None, p=None,
        title="", filename=None, lims=True):

    # plt.ioff()
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 10)) # Tutaj sie psuje!!!
    ax = plt.gca()
    ax.set_aspect('equal')

    if lims:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

    color_arr = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'lime',
                 'purple', 'pink', 'brown', 'orange', 'teal', 'coral',
                 'lightblue', 'black']
    color_arr = np.array(color_arr)



    if "train" in filename:
        must_z = []
        cannot_z = []
        for idx, link in enumerate(dataset.links[:500]):
            if link[1]:
                must_z += [z[2 * idx:2 * idx + 2]]
            else:
                cannot_z += [z[2 * idx:2 * idx + 2]]

        must_z = np.array(must_z)
        cannot_z = np.array(cannot_z)
        # print(must_z.shape, cannot_z.shape)

        for must_pair in must_z:
            plt.scatter(must_pair[:, 0], must_pair[:, 1], c="green")
            plt.plot(must_pair[:, 0], must_pair[:, 1], c="blue")
        for cannot_pair in cannot_z:
            plt.scatter(cannot_pair[:, 0], cannot_pair[:, 1], c="purple")
            plt.plot(cannot_pair[:, 0], cannot_pair[:, 1], c="red")

    else:
        labeled_z = z[y.sum(1) == 1]
        unlabeled_z = z[y.sum(1) == 0]
        p = p / p.sum()
        if len(unlabeled_z) > 0:
            plt.scatter(unlabeled_z[:, 0], unlabeled_z[:, 1], c="gray")

        plt.scatter(labeled_z[:, 0], labeled_z[:, 1], c=color_arr[np.argmax(y, axis=1)])

    if means is not None:
        for i in range(len(means)):
            c = color_arr[i]
            circle = plt.Circle(
                    means[i], alpha[i],
                    color=c, fill=False, linewidth=p[i] * 30)
            ax.add_artist(circle)
            plt.scatter(means[i, 0], means[i, 1],
                    s=0 + (p[i] * alpha[i] * 1000),
                    c=c, label=dataset.labels_names[i], marker="^")

    plt.title(title)
    plt.legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    plt.close()


def evaluate_model(
        sess, model, valid_set, epoch, dataset,
        filename_prefix="test", subset=None,
        class_in_sum=True, training_mode=False):

    if len(valid_set["X"]) == 0:
        return None, None

    means_val, alphas_val, p_val = sess.run(
            [model.gausses["means"],
             model.gausses["variations"],
             model.gausses["probs"]])

    cost_sum = (model.costs["full"] if class_in_sum
                else model.costs["unsupervised"])
    metrics_tensors = {
        "cramer-wold": model.costs["cw"],
        "reconstruction": model.costs["reconstruction"],
        # "classification": model.costs["class"],
        "distance": model.costs["distance"],
        "erf": model.costs["erf"],
        "cec": model.costs["cec"],
        "link": model.costs["link"],
        "sum": cost_sum
    }

    metrics_final = {k:[] for k in metrics_tensors.keys()}
    metrics_final["accuracy"] = []

    if subset != None:
        valid_set = {"X": valid_set["X"][:subset],
                     "y": valid_set["y"][:subset]}

    batch_size = 500
    batch_num = int(np.ceil(len(valid_set["X"]) / batch_size))
    costs_arr = []
    all_z = []
    preds = []
    for batch_idx in range(batch_num):

        X_batch = valid_set["X"][batch_idx * batch_size:(batch_idx + 1) * batch_size]
        y_batch = valid_set["y"][batch_idx * batch_size:(batch_idx + 1) * batch_size]

        feed_dict = {
                model.placeholders["X"]: X_batch,
                model.placeholders["X_target"]: X_batch,
                model.placeholders["y"]: y_batch,
                model.placeholders["must_link"]: np.zeros((len(X_batch),)),
                model.placeholders["cannot_link"]: np.zeros((len(X_batch),)),
                model.placeholders["training"]: training_mode}

        metrics, class_logits, out_z = sess.run(
                [metrics_tensors, model.preds, model.out["z"]],
                feed_dict=feed_dict)

        # metrics["classification"] *= y_batch.sum()
        pred = class_logits.argmax(-1)
        preds += pred.tolist()

        correct_n = np.sum((pred == y_batch.argmax(-1))
                           * y_batch.sum(axis=-1))

        metrics["accuracy"] = correct_n
        all_z += [out_z]

        for key, value in metrics.items():
            metrics_final[key] += [value]


    all_z = np.vstack(all_z)
    preds = np.array(preds)
    emp_variances = []
    for label in range(dataset.classes_num):
        labeled_points = all_z[preds == label]
        emp_var = np.std(labeled_points, axis=0)
        emp_var = np.nan_to_num(emp_var)
        # print(emp_var, len(emp_var))
        emp_variances += [np.mean(np.square(emp_var))]

    rand_score = adjusted_rand_score(valid_set["y"].argmax(-1), preds)
    metrics_final["rand_score"] = rand_score
    metrics_final["nmi"] = normalized_mutual_info_score(
        valid_set["y"].argmax(-1), preds, average_method="geometric")

    # PCA
    if epoch % 5 == 0:
        if model.z_dim == 2: # If we're on a plane, PCA is not necessary
            pca_results = all_z
            pca_means = means_val
        else:
            pca = PCA(2)
            pca_results = pca.fit_transform(all_z)
            pca_means = pca.transform(means_val)

        graph_filename = "results/{}/graphs/{}_epoch_{}.png".format(
                model.name, filename_prefix, str(epoch).zfill(3))
        draw_gmm(pca_results, valid_set["y"], dataset,
                 pca_means, alphas_val, p_val, lims=False,
                 filename=graph_filename)
    # T-SNE
    # if epoch % 10 == 0:
    #     tsne = TSNE(2)
    #     tsne_results = tsne.fit_transform(all_z[:1000])
    #     draw_gmm(tsne_results, valid_set["y"], lims=False,
    #              filename=graphs_folder + "/tsne_{}_epoch_{}.png".format(filename_prefix, str(epoch).zfill(3)))

    for key, value in metrics_final.items():
        if key is "accuracy" or key is "classification":
            metrics_final[key] = np.sum(value) / valid_set["y"].sum()
        else:
            metrics_final[key] = np.mean(value)

    if epoch % 5 == 0:
        print(epoch, "Dataset:", filename_prefix, end="\t")
        for key in ["nmi", "rand_score", "cramer-wold", "reconstruction", "link", "distance"]:
            print("{}: {:.4f}".format(key, metrics_final[key]), end=" ")
        print()

    return metrics_final, emp_variances


def sample_from_classes(sess, model, dataset, epoch, valid_var=None, show_only=False):
    means_val, alphas_val = sess.run(
            [model.gausses["means"], model.gausses["variations"]])
    im_h, im_w, im_c = dataset.image_shape

    if valid_var is not None:
        alphas_val = valid_var
        print("Empirical variances", valid_var)
    else:
        print("Analytic variances", alphas_val)

    canvas = np.empty((im_h * dataset.classes_num, im_w * 10, im_c))
    for row_idx, (mean, cov) in enumerate(zip(means_val, alphas_val)):
        dim = len(mean)
        samples = np.random.multivariate_normal(mean, cov * np.eye(dim), size=10)
        generated = sess.run(model.out["y"], {model.out["z"]: samples})

        for col_idx, sample in enumerate(generated):
            if dataset.whitened:
                sample = dataset.blackening(sample)
            # sample = 1 / (1 + np.exp(-sample))  # Sigmoid
            sample[sample < 0] = 0
            sample[sample > 1] = 1
            sample = sample.reshape(dataset.image_shape)

            canvas[row_idx * im_h:(row_idx + 1) * im_h,
                   col_idx * im_w:(col_idx + 1) * im_w] = sample

        plt.ylabel(
                dataset.labels_names[row_idx], fontsize=15, rotation=0.45)
        plt.yticks([])

    # fig.tight_layout(pad=0)
    plt.imshow(canvas.squeeze(), cmap="gray")
    if show_only:
        plt.show()
    else:
        if valid_var is not None:
            filename = "results/{}/empvar_sampling_{}.png".format(
                    model.name, str(epoch).zfill(3))
        else:
            filename = "results/{}/sampling_{}.png".format(
                    model.name, str(epoch).zfill(3))
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def inter_class_interpolation(sess, model, dataset, epoch, show_only=False):
    pass


def interpolation(sess, model, dataset, epoch, show_only=False):
    out_z=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[15,17],[18,1]]

    nx = 10
    ny = 10
    n_iterations = nx

    x_values = np.linspace(-2, 2, nx)
    y_values = np.linspace(-2, 2, ny)

    im_h, im_w, im_c = dataset.image_shape
    # im_w = 28
    # im_h = 28
    # im_c = 1 # channels

    canvas = np.empty((im_h * ny, im_w * nx, im_c))
    for i, yi in enumerate(y_values):
        x_sample = dataset.test["X"][out_z[i]]

        out_z1 = sess.run(
                model.out["z"],
                {model.placeholders["X"]: x_sample})
        #self.model.encode(x_sample)
        codings_rnd = out_z1[0]
        target_codings = out_z1[1]

        A = np.array(codings_rnd)
        for iteration in np.arange(1,n_iterations):
            codings_interpolate = codings_rnd + (target_codings - codings_rnd) * iteration / n_iterations
            A = np.vstack([A,codings_interpolate])


        y1 = sess.run(model.out["y"], {model.out["z"]: A})

        for j, xi in enumerate(x_values):
            d_plot = y1[j]
            # d_plot /= 255
            # d_plot = 1 - d_plot
            # d_plot *= 255
            if dataset.whitened:
                d_plot = dataset.blackening(d_plot)

            # d_plot = 1 / (1 + np.exp(-d_plot))
            d_plot[d_plot<0] = 0
            d_plot[d_plot>1] = 1
            canvas[(ny-i-1)* im_h:(ny-i) * im_h, j * im_w:(j+1)* im_w] = d_plot.reshape(im_h, im_w, im_c)

    canvas = canvas.squeeze() # (28, 28, 1) => (28, 28)
    plt.figure(figsize=(nx, ny))
    plt.axes().set_aspect('equal')
    plt.axis("off")
    plt.imshow(canvas, origin="upper", cmap="gray")
    # plt.tight_layout(pad=0)
    if show_only:
        plt.show()
    else:
        filename = "results/{}/interpolation_{}.png".format(
            model.name, str(epoch).zfill(3))
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_costs(fig, costs, name):
    ax = fig.axes
    first_time = False
    if len(ax) == 1:
        ax_1 = ax[0]
        ax_2 = ax_1.twinx()
        first_time = True
    else:
        ax_1, ax_2 = ax

    ax_1.clear()
    ax_2.clear()
    ax_1.set_title(name)

    for key in ["reconstruction", "cec", "distance"]:
        ax_1.plot(costs[key], label=key)

    for key in ["rand_score", "nmi"]:
        ax_2.plot(costs[key], label=key, c="red")

    if first_time:
        fig.legend(loc=3)
    fig.canvas.draw()

def save_costs(model, costs, dataset_type):
    cost_dict = {k: [] for k in costs[0].keys()}
    for cost in costs:
        for key, value in cost.items():
            cost_dict[key] += [value]

    fig, _ = plt.subplots()
    plot_costs(fig, cost_dict, dataset_type)
    results_dir = "results/{}".format(model.name)
    fig.savefig(results_dir + "/{}_losses.png".format(dataset_type))

    costs_arr = list((key, val) for key, val in cost_dict.items())

    np.savetxt(
        results_dir + "/{}_log.txt".format(dataset_type),
        np.array(list(c[1] for c in costs_arr)).T,
        fmt="%10.5f",
        header=" ".join(list(c[0] for c in costs_arr)))
