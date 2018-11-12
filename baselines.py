import numpy as np
import tensorflow as tf
from tqdm import trange

frugal_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

# Convolutional baseline
def baseline_cnn_classifier(x, dataset):
    im_h, im_w, im_c = dataset.image_shape
    h = x
    h = tf.reshape(h, (-1, im_h, im_w, im_c))
    h = tf.layers.conv2d(h, 64, 3, strides=1, padding="same", activation=tf.nn.relu)
    h = tf.layers.conv2d(h, 64, 3, strides=2, padding="same", activation=tf.nn.relu)
    h = tf.layers.conv2d(h, 64, 3, strides=2, padding="same", activation=tf.nn.relu)

    h = tf.layers.flatten(h)
    h = tf.layers.dense(h, units=128, name='fc_1')
    y_pred = tf.layers.dense(h, units=dataset.classes_num, name='baseline_z_mean')

    return y_pred

def baseline_fn_classifier(x, dataset):
    h = x
    h = tf.layers.dense(h, units=128, activation=tf.nn.relu)
    h = tf.layers.dense(h, units=128, activation=tf.nn.relu)
    h = tf.layers.dense(h, units=128, activation=tf.nn.relu)
    y_pred = tf.layers.dense(h, units=dataset.classes_num, name='baseline_y_mean')

    return y_pred

def build_classifier(model_type, dataset):
    with tf.variable_scope("baseline_{}".format(model_type), reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(1e-3)
        baseline_x = tf.placeholder(shape=np.append([None], dataset.x_dim), dtype=tf.float32, name='input_x')
        baseline_target = tf.placeholder(shape=[None, dataset.classes_num], dtype=tf.float32, name='target_y')

        if model_type == "cnn":
            baseline_y = baseline_cnn_classifier(baseline_x, dataset)
        elif model_type == "fn":
            baseline_y = baseline_fn_classifier(baseline_x, dataset)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=baseline_y, labels=baseline_target))
        train_op = optimizer.minimize(loss)
        return baseline_x, baseline_y, baseline_target, train_op


def train_classifier(model_type, dataset):
    batch_size = 16
    n_epochs = 101

    dataset_baseline = dataset.labeled_train["X"], dataset.labeled_train["y"]
    with tf.Session(config=frugal_config) as sess:
        baseline_x, baseline_y, baseline_target, baseline_train_op = build_classifier(model_type, dataset)
        sess.run(tf.global_variables_initializer())
        train_examples_num = dataset_baseline[0].shape[0]
        batches_num = train_examples_num // batch_size

        for epoch in trange(n_epochs):
            for batch_index in trange(batches_num, leave=False):
                X_batch = dataset_baseline[0][batch_index * batch_size:(batch_index + 1) * batch_size]
                y_batch = dataset_baseline[1][batch_index * batch_size:(batch_index + 1) * batch_size]

                sess.run(baseline_train_op, feed_dict={baseline_x: X_batch, baseline_target: y_batch})

            if epoch % 10 == 0:
                X_batch = dataset.test["X"][:1000]
                y_batch = dataset.test["y"][:1000]

                y_pred_val = sess.run(baseline_y, feed_dict={baseline_x: X_batch, baseline_target: y_batch})
                # print(np.argmax(y_pred_val, axis=-1))
                acc = np.sum(np.argmax(y_pred_val, axis=-1) == np.argmax(y_batch, axis=-1)) / 1000
                print(acc)

        batch_size = 1000
        acc_counts = []
        batch_num = len(dataset.test["X"]) // batch_size
        for batch_idx in trange(batch_num):
            X_batch = dataset.test["X"][batch_idx * batch_size:(batch_idx + 1) * batch_size]
            y_batch = dataset.test["y"][batch_idx * batch_size:(batch_idx + 1) * batch_size]
            cp = sess.run(baseline_y, {baseline_x: X_batch })
            acc_count = np.count_nonzero(cp.argmax(axis=-1) == y_batch.argmax(axis=-1))
            acc_counts += [acc_count / batch_size]

        full_acc = np.mean(acc_counts)
        acc_counts = np.hstack((acc_counts, full_acc))
        print("Srednia", np.mean(acc_counts))
        header = "{} results over {} examples and overall".format(batch_size, batch_num)
        print(acc_counts)
        np.savetxt(
            results_folder + "/baseline_{}_log.txt".format(model_type),
            acc_counts, fmt="%10.5f", header=header)
    # print("\r{}%".format(100 * batch_index //  (mnist.train.num_examples // batch_size) ), end="")

# tf.reset_default_graph()
# train_classifier("cnn")
# tf.reset_default_graph()
# train_classifier("fn")
