import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from tqdm import trange
from PIL import Image

plt.switch_backend("agg")


def draw_gmm(
        z, y, dataset, means=None, alpha=None, p=None,
        title="", filename=None, lims=True):

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect('equal')

    if lims:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

    color_arr = [
        'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'lime', 'purple',
        'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'black'
        ]
    color_arr = np.array(color_arr)

    labeled_z = z[y.sum(1) == 1]
    unlabeled_z = z[y.sum(1) == 0]
    p = p / p.sum()

    if len(unlabeled_z) > 0:
        plt.scatter(unlabeled_z[:, 0], unlabeled_z[:, 1], c="gray")

    c = color_arr[np.argmax(y, axis=1)[:len(labeled_z)]]

    plt.scatter(labeled_z[:, 0], labeled_z[:, 1], c=c)

    # TODO: nie rysujemy meanow
    if means is not None:
        for i in range(len(means)):
            c = color_arr[i]
            plt.scatter(
                means[i, 0], means[i, 1], c=c,
                label=dataset.labels_names[i]
                )

    plt.title(title)
    plt.legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)


def evaluate_gmmcwae(
        sess, model, valid_set, epoch, dataset,
        filename_prefix="test", subset=None,
        class_in_sum=True, training_mode=False):

    means_val, alphas_val, p_val = sess.run(
            [model.gausses["means"],
             model.gausses["variations"],
             model.gausses["probs"]])

    cost_sum = (model.costs["full"] if class_in_sum
                else model.costs["unsupervised"])
    metrics_tensors = {
        "cramer-wold": model.costs["cw"],
        "reconstruction": model.costs["reconstruction"],
        "classification": model.costs["class"],
        "distance": model.costs["distance"],
        "sum": cost_sum
    }

    metrics_final = {k: [] for k in metrics_tensors.keys()}
    metrics_final["accuracy"] = []

    if subset is not None:
        valid_set = {"X": valid_set["X"][:subset],
                     "y": valid_set["y"][:subset]}

    batch_size = 500
    batch_num = int(np.ceil(len(valid_set["X"]) / batch_size))
    all_z = []
    preds = []
    for batch_idx in range(batch_num):
        X_batch = valid_set["X"][batch_idx * batch_size:(batch_idx + 1) * batch_size]
        y_batch = valid_set["y"][batch_idx * batch_size:(batch_idx + 1) * batch_size]

        feed_dict = {
                model.placeholders["X"]: X_batch,
                model.placeholders["y"]: y_batch,
                model.placeholders["training"]: training_mode}

        metrics, class_logits, out_z = sess.run(
                [metrics_tensors, model.preds, model.out["z"]],
                feed_dict=feed_dict)

        metrics["classification"] *= y_batch.sum()
        pred = class_logits.argmax(-1)
        preds += pred.tolist()

        correct_n = np.sum((pred == y_batch.argmax(-1)) * y_batch.sum(axis=-1))

        metrics["accuracy"] = correct_n
        all_z += [out_z]

        for key, value in metrics.items():
            metrics_final[key] += [value]

    all_z = np.vstack(all_z)
    preds = np.array(preds)
    emp_variances = []
    emp_means = []
    for label in range(dataset.classes_num):
        labeled_points = all_z[preds == label]
        if labeled_points.size == 0: 
            emp_means += [model.z_dim * [float("nan")]]
            emp_variances += [model.z_dim * [0]]
            continue
        emp_means += [np.mean(labeled_points, axis=0)]
        emp_var = np.std(labeled_points, axis=0)
        emp_var = np.nan_to_num(emp_var)
        # print(emp_var, len(emp_var))
        emp_variances += [np.mean(np.square(emp_var))]
    emp_means = np.array(emp_means)
    emp_variances = np.array(emp_variances)

    rand_score = adjusted_rand_score(preds, valid_set["y"].argmax(-1))
    metrics_final["rand_score"] = rand_score

    # PCA
    if epoch % 5 == 0:
        try:
            if model.z_dim == 2:  # If we're on a plane, PCA is not necessary
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
        except ValueError as e:
            print(e)

    for key, value in metrics_final.items():
        if key == "accuracy" or key == "classification":
            metrics_final[key] = np.sum(value) / valid_set["y"].sum()
        else:
            metrics_final[key] = np.mean(value)

    if epoch % 5 == 0:
        print(epoch, "Dataset:", filename_prefix, end="\t")
        metrics_list = [
            "accuracy", "cramer-wold", "reconstruction",
            "distance", "classification"
            ]
        for key in metrics_list:
            print("{}: {:.4f}".format(key, metrics_final[key]), end=" ")
        print()

    return metrics_final, emp_variances, emp_means


def save_distance_matrix(sess, model, epoch):
    means_val, alphas_val = sess.run(
           [model.gausses["means"], model.gausses["variations"]])
    distance_matrix = np.expand_dims(means_val, 0) - np.expand_dims(means_val, 1)
    distance_matrix = np.sqrt(np.sum(np.square(distance_matrix), -1))

    filename = "results/{}/distances.txt".format(model.name)

    with open(filename, "a") as f:
        f.write("Epoch {}\n".format(epoch))
        for line in distance_matrix:
            f.write("\t".join([str(number) for number in line]) + "\n")


def sample_from_classes(
        sess, model, dataset, epoch,
        valid_var=None, show_only=False):
    means_val, alphas_val = sess.run(
            [model.gausses["means"], model.gausses["variations"]])

    if means_val.shape[0] == 1:
        means_val = np.tile(means_val, (10, 1))
        alphas_val = np.tile(alphas_val, (10,))

    if dataset.name == "svhn":
        means_val = np.roll(means_val, 1, axis=0)
        alphas_val = np.roll(alphas_val, 1, axis=0)

    im_h, im_w, im_c = dataset.image_shape

    canvas = np.empty((im_h * len(means_val), im_w * 10, im_c))

    dim = len(means_val[0])
    samples = np.random.multivariate_normal(
        [0.] * dim, np.eye(dim), size=10)
    for row_idx, (mean, cov) in enumerate(zip(means_val, alphas_val)):

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

    plt.xticks([])
    plt.yticks([])

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


def save_interpolation_samples(sess, model, dataset, n_samples, from_dataset=True):
    im_h, im_w, im_c = dataset.image_shape
    batch_size = 1000
    batch_num = (n_samples // batch_size)

    if not from_dataset:
        output_tensors = [
            model.gausses["means"],
            model.gausses["variations"],
            model.gausses["probs"]
        ]

        mean_vals, cov_vals, prob_vals = sess.run(output_tensors)

    assert batch_num * batch_size == n_samples

    for batch_idx in range(batch_num):
        if from_dataset:
            indices = np.random.choice(
                len(dataset.test["X"]),
                replace=False,
                size=(batch_size * 2)
            )
            samples = sess.run(
                model.out["z"],
                {model.placeholders["X"]: dataset.test["X"][indices]}
            )
            samples = samples.reshape((batch_size, 2, -1))
        else:
            samples = np.random.normal(size=(batch_size * 2, model.z_dim))  # [N, C]
            class_indices = np.random.choice(
                dataset.classes_num,  size=batch_size * 2, p=prob_vals,
                )  # [N]
            samples += mean_vals[class_indices]
            samples = samples.reshape((batch_size, 2, -1))

        interpols = (samples[:, 1] - samples[:, 0]) * np.random.random((batch_size, 1))
        interpols = interpols + samples[:, 0]
        interpols.reshape(batch_size, -1)
        generated = sess.run(model.out["y"], {model.out["z"]: interpols})
        if dataset.whitened:
            generated = dataset.blackening(generated)

        generated[generated < 0] = 0
        generated[generated > 1] = 1
        generated = (generated * 255).astype("uint8")
        generated = generated.reshape([-1] + dataset.image_shape)

        for idx, pixels in enumerate(generated):
            postfix = "" if from_dataset else "_gen"
            filename = "results/{}/final_inter{}_samples/inter{}_{}.png".format(
                model.name, postfix, str(batch_idx).zfill(2), str(idx).zfill(5))
            if dataset.name == "mnist":
                img = Image.fromarray(pixels.squeeze(), "L")
            else:
                img = Image.fromarray(pixels, "RGB")
            img.save(filename)


def save_samples(sess, model, dataset, n_samples):
    means_val, covs_val, prob_vals = sess.run(
        [model.gausses["means"], model.gausses["variations"], model.gausses["probs"]]
        )
    im_h, im_w, im_c = dataset.image_shape

    if len(means_val) == 1:
        means_val = np.tile(means_val, (dataset.classes_num, 1))
        covs_val = np.tile(covs_val, (dataset.classes_num,))

    G = len(means_val)

    samples_per_class = np.random.multinomial(
        10000, prob_vals)
    print(samples_per_class)

    for class_idx in trange(G):
        mean = means_val[class_idx]
        cov = covs_val[class_idx]

        samples = np.random.multivariate_normal(
            mean, cov * np.eye(len(mean)), size=samples_per_class[class_idx])
        generated = sess.run(model.out["y"], {model.out["z"]: samples})

        if dataset.whitened:
            generated = dataset.blackening(generated)
        generated[generated < 0] = 0
        generated[generated > 1] = 1
        generated = (generated * 255).astype("uint8")
        generated = generated.reshape([-1] + dataset.image_shape)

        for idx, pixels in enumerate(generated):
            filename = "results/{}/final_samples/c{}_{}.png".format(
                model.name, class_idx, str(idx).zfill(5))
            if dataset.name == "mnist":
                img = Image.fromarray(pixels.squeeze(), "L")
            else:
                img = Image.fromarray(pixels, "RGB")
            img.save(filename)


def inter_class_interpolation(
        sess, model, dataset, epoch,
        show_only=False, extrapolate=False):
    means_val, alphas_val, p_val = sess.run(
            [model.gausses["means"],
             model.gausses["variations"],
             model.gausses["probs"]])

    mean_diff_matrix = np.expand_dims(means_val, 0) - np.expand_dims(means_val, 1)
    im_h, im_w, im_c = dataset.image_shape

    samples_num = 10
    interpolation_steps = 10
    canvas = np.zeros((im_h * samples_num, im_w * (interpolation_steps + 2), im_c))

    interpolating_samples = list(range(samples_num))
    tensor_z = sess.run(
        model.out["z"],
        {model.placeholders["X"]: dataset.test["X"][interpolating_samples]}
    )

    for idx in interpolating_samples:
        label = np.argmax(dataset.test["y"][idx])
        interpolation_direction = (
            mean_diff_matrix[label][(label + 1) % dataset.classes_num])

        z_inputs = []
        linspace_end = -0.5 if extrapolate else 1.
        for step_size in np.linspace(0, linspace_end, num=interpolation_steps + 2):
            z_inputs += [tensor_z[idx] + interpolation_direction * step_size]

        outputs = sess.run(model.out["y"], {model.out["z"]: z_inputs})

        for output_idx, output in enumerate(outputs):
            if dataset.whitened:
                output = dataset.blackening(output)

            canvas[im_h * idx:im_h * (idx + 1),
                   im_w * output_idx:im_w * (output_idx + 1)] = output.reshape(im_h, im_w, im_c)

    fig = plt.figure(figsize=(interpolation_steps + 2, samples_num))

    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.axis("off")
    plt.imshow(canvas.squeeze(), origin="upper", cmap="gray")
    if show_only:
        plt.show()
    else:
        interpolation_type = "extrapolation" if extrapolate else "interpolation"
        filename = "results/{}/class_{}_{}.png".format(
            model.name, interpolation_type, str(epoch).zfill(3))
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def reconstruction(input_indices, sess, model, dataset, epoch):
    im_h, im_w, im_c = dataset.image_shape

    originals = dataset.test["X"][input_indices]
    outputs = sess.run(
        model.out["y"],
        {model.placeholders["X"]: dataset.test["X"][input_indices]}
    )

    for output_idx, output in enumerate(outputs):
        if dataset.whitened:
            output = dataset.blackening(output)

        output = output.reshape(im_h, im_w, im_c)
        original = originals[output_idx].reshape(im_h, im_w, im_c)
        whole_image = np.concatenate([output, original], 1).squeeze()
        plt.imshow(whole_image, cmap="gray")
        filename = "results/{}/reconstruction_{}_{}.png".format(
            model.name, str(epoch).zfill(3), output_idx)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)


def chosen_class_interpolation(
        input_indices, sess, model, dataset, epoch,
        show_only=False, direct=False, extrapolate=False, chosen_inters=None):

    means_val, alphas_val, p_val = sess.run(
            [model.gausses["means"],
             model.gausses["variations"],
             model.gausses["probs"]])

    if dataset.name == "svhn":
        # Move the "zeros" row from the end to the beginning
        means_val = np.roll(means_val, 1, axis=0)
        alphas_val = np.roll(alphas_val, 1, axis=0)
        dataset.label_names = [str(idx) for idx in range(10)]

    mean_diff_matrix = np.expand_dims(means_val, 0) - np.expand_dims(means_val, 1)
    im_h, im_w, im_c = dataset.image_shape

    samples_num = len(input_indices)
    interpolation_steps = 10
    padding = 4

    tensor_z, probs = sess.run(
        [model.out["z"], model.out["probs"]],
        {model.placeholders["X"]: dataset.test["X"][input_indices]}
    )

    inter_type = "direct" if direct else "cyclic"
    inter_direction = "extrapolation" if extrapolate else "interpolation"

    for idx in range(samples_num):
        fig = plt.figure(figsize=(interpolation_steps, 1))

        label = np.argmax(probs[idx])
        if dataset.name == "svhn":
            label = (label + 1) % 10

        canvas = np.ones((im_h + padding, im_w * interpolation_steps, im_c))
        direction = chosen_inters[idx]
        if direct:
            interpolation_direction = means_val[direction] - tensor_z[idx]
        else:
            interpolation_direction = mean_diff_matrix[label][direction]

        linspace_end = -0.5 if extrapolate else 1.
        z_inputs = []
        for step_size in np.linspace(0., linspace_end, num=interpolation_steps):
            z_inputs += [tensor_z[idx] + interpolation_direction * step_size]

        outputs, sample_probs = sess.run(
            [model.out["y"], model.out["probs"]],
            {model.out["z"]: z_inputs}
        )

        for output_idx, output in enumerate(outputs):
            if dataset.whitened:
                output = dataset.blackening(output)
            caption = "{} {}%".format(
                dataset.labels_names[sample_probs[output_idx].argmax()],
                round(float(sample_probs[output_idx].max() * 100), 2))
            plt.text(im_w * output_idx, 3, caption, fontsize=6)

            canvas[padding:im_h + padding,
                   im_w * output_idx:im_w * (output_idx + 1)] = output.reshape(im_h, im_w, im_c)

        plt.xticks([])
        plt.yticks([])
        plt.axis('equal')
        plt.axis("off")
        plt.imshow(canvas.squeeze(), origin="upper", cmap="gray")
        filename = "results/{}/{}_{}_{}-{}.png".format(
            model.name, inter_type, inter_direction, input_indices[idx], direction)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def cyclic_interpolation(
        input_indices, sess, model, dataset, epoch,
        show_only=False, direct=False, extrapolate=False):

    means_val, alphas_val, p_val = sess.run(
            [model.gausses["means"],
             model.gausses["variations"],
             model.gausses["probs"]])

    if dataset.name == "svhn":
        # Move the "zeros" row from the end to the beginning
        means_val = np.roll(means_val, 1, axis=0)
        alphas_val = np.roll(alphas_val, 1, axis=0)
        dataset.label_names = [str(idx) for idx in range(10)]

    mean_diff_matrix = np.expand_dims(means_val, 0) - np.expand_dims(means_val, 1)
    im_h, im_w, im_c = dataset.image_shape

    samples_num = len(input_indices)
    interpolation_steps = 10
    padding = 6

    tensor_z, probs = sess.run(
        [model.out["z"], model.out["probs"]],
        {model.placeholders["X"]: dataset.test["X"][input_indices]}
    )

    for idx in range(samples_num):
        fig = plt.figure(figsize=(interpolation_steps, samples_num))

        # label = np.argmax(dataset.test["y"][idx])
        label = np.argmax(probs[idx])
        if dataset.name == "svhn":
            label = (label + 1) % 10

        canvas = np.ones((
            (im_h + padding) * (dataset.classes_num),
            im_w * interpolation_steps,
            im_c
            ))

        for direction in range(dataset.classes_num):

            if direct:
                interpolation_direction = means_val[direction] - tensor_z[idx]
            else:
                interpolation_direction = mean_diff_matrix[label][direction]

            linspace_end = -0.5 if extrapolate else 1.
            z_inputs = []
            for step_size in np.linspace(0., linspace_end, num=interpolation_steps):
                z_inputs += [tensor_z[idx] + interpolation_direction * step_size]

            outputs, sample_probs = sess.run(
                [model.out["y"], model.out["probs"]],
                {model.out["z"]: z_inputs}
            )

            for output_idx, output in enumerate(outputs):
                if dataset.whitened:
                    output = dataset.blackening(output)
                start_h = (im_h + padding) * direction
                caption = "{} {}%".format(
                    dataset.labels_names[sample_probs[output_idx].argmax()],
                    round(float(sample_probs[output_idx].max() * 100), 2))
                plt.text(im_w * output_idx, start_h - 2, caption, fontsize=6)

                canvas[start_h:start_h + im_h,
                       im_w * output_idx:im_w * (output_idx + 1)] = output.reshape(im_h, im_w, im_c)

        inter_type = "direct" if direct else "cyclic"
        inter_direction = "extrapolation" if extrapolate else "interpolation"

        plt.xticks([])
        plt.yticks([])
        plt.axis('equal')
        plt.axis("off")
        plt.imshow(canvas.squeeze(), origin="upper", cmap="gray")
        plt.tight_layout(pad=0)
        filename = "results/{}/{}_{}_{}_{}.png".format(
            model.name, inter_type, inter_direction,
            str(epoch).zfill(3), input_indices[idx])
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def interpolation(input_indices, sess, model, dataset, epoch, separate_files=False):
    out_z = [
        [input_indices[idx], input_indices[idx + 1]]
        for idx in range(len(input_indices) - 1)
    ]
    out_z += [[input_indices[-1], input_indices[0]]]
    out_z.reverse()

    # TODO: przejrzyj to

    n_iterations = 8
    nx = n_iterations + 2
    ny = len(out_z)

    im_h, im_w, im_c = dataset.image_shape
    # im_w = 28
    # im_h = 28
    # im_c = 1 # channels

    canvas = np.empty((im_h * ny, im_w * (nx), im_c))

    for y_idx in range(ny):
        x_sample = dataset.test["X"][out_z[y_idx]]
        y_sample = dataset.test["y"][out_z[y_idx]]

        one_hot_class = [0.] * 10
        one_hot_class[0] = 1.

        feed_dict = {
            model.placeholders["X"]: x_sample,
            model.placeholders["y"]: np.zeros((len(x_sample), dataset.classes_num)),
            # model.out["simplex_y"]: one_hot_class,
            # model.out["probs"]: [one_hot_class] * len(x_sample)
        }
        # TODO: brudny hack
        out_z1 = sess.run(model.out["z"], feed_dict)
        codings_rnd = out_z1[0]
        target_codings = out_z1[1]

        A = np.array(codings_rnd)
        B = np.array(y_sample[0])
        for iteration in np.arange(1, n_iterations):
            step_direction = (target_codings - codings_rnd)
            step = step_direction * iteration / n_iterations
            codings_interpolate = codings_rnd + step
            y_interpolate = y_sample[0] + (y_sample[1] - y_sample[0]) * iteration / n_iterations
            B = np.vstack([B, y_interpolate])
            A = np.vstack([A, codings_interpolate])

        y1 = sess.run(model.out["y"], {
            model.out["z"]: A,
            # model.out["simplex_y"]: one_hot_class,
            # model.out["probs"]: B
            }
        )

        y1 = np.concatenate(
            (x_sample[0].reshape(1, -1), y1, x_sample[1].reshape(1, -1)), axis=0)
        for x_idx, d_plot in enumerate(y1):
            if dataset.whitened:
                d_plot = dataset.blackening(d_plot)

            # d_plot = 1 / (1 + np.exp(-d_plot))
            d_plot[d_plot < 0] = 0
            d_plot[d_plot > 1] = 1

            canvas[(ny - y_idx - 1) * im_h:(ny - y_idx) * im_h,
                   x_idx * im_w:(x_idx + 1) * im_w] = d_plot.reshape(im_h, im_w, im_c)

    canvas = canvas.squeeze()  # (28, 28, 1) => (28, 28)
    fig = plt.figure(figsize=(nx, ny))

    if separate_files:
        for idx in range(ny):
            plt.xticks([])
            plt.yticks([])
            plt.axis('equal')
            plt.axis("off")

            row = canvas[idx * im_h:(idx + 1) * im_h]
            plt.imshow(row, origin="upper", cmap="gray")
            filename = "results/{}/interpolation_{}.png".format(
                model.name, input_indices[idx])
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
    else:
        plt.xticks([])
        plt.yticks([])
        plt.axis('equal')
        plt.axis("off")
        plt.imshow(canvas, origin="upper", cmap="gray")
        filename = "results/{}/interpolation_{}.png".format(
            model.name, str(epoch).zfill(3))
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


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

    for key in ["classification", "cramer-wold", "reconstruction"]:
        if key in costs:
            ax_1.plot(costs[key], label=key)

    for key in ["accuracy"]:
        if key in costs:
            ax_2.set_ylim(0, 1)
            ax_2.plot(costs[key], label=key, c="red")

    plt.title("Metrics")
    ax_1.set_xlabel("Epoch")
    if first_time:
        fig.legend(loc=3)
    fig.canvas.draw()
    plt.close(fig)


def save_costs(model, costs, dataset_type):
    cost_dict = {k: [] for k in costs[0].keys()}
    for cost in costs:
        for key, value in cost.items():
            cost_dict[key] += [value]

    fig, _ = plt.subplots()
    plot_costs(fig, cost_dict, dataset_type)
    results_dir = "results/{}".format(model.name)
    fig.savefig(results_dir + "/{}_losses.png".format(dataset_type), dpi=300)
    plt.close(fig)

    costs_arr = list((key, val) for key, val in cost_dict.items())

    np.savetxt(
        results_dir + "/{}_log.txt".format(dataset_type),
        np.array(list(c[1] for c in costs_arr)).T,
        fmt="%10.5f",
        header=" ".join(list(c[0] for c in costs_arr)))
