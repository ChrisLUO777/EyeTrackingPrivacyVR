import itertools
from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf
from keras.models import load_model

from JFGComparatorTrain import jfg_comparison_generator
from JointFeatureGeneratorTrain import gaze_map_generator


def generate_pairs(x_batch, y_batch):
    pairs = list(itertools.combinations(range(len(x_batch)), 2))
    x1 = []
    x2 = []
    y_true = []

    for i, j in pairs:
        x1.append(x_batch[i])
        x2.append(x_batch[j])
        y_true.append(int(y_batch[i] == y_batch[j]))

    x1 = tf.stack(x1)
    x2 = tf.stack(x2)
    y_true = tf.expand_dims(tf.cast(y_true, tf.float32), axis=-1)

    return x1, x2, y_true


def generate_sim_scores_and_labels(model, dataset):
    similarity_scores = []
    labels = []
    for f1, f2, y_batch in dataset:

        # Compute the similarity score using the comparator
        y_pred = model([f1, f2], training=False)
        # y_pred = 0.0 - np.sqrt(np.sum((f1 - f2) ** 2, axis=1))

        similarity_scores.extend(y_pred)
        labels.extend(y_batch)

    return similarity_scores, labels


def evaluate_threshold(y_true, y_pred, threshold):
    binary_pred = y_pred > threshold
    accuracy = f1_score(y_true, binary_pred)
    # accuracy = np.mean(binary_pred == y_true)
    return accuracy


def find_best_threshold(y_true, y_pred, num_thresholds=200):
    thresholds = np.linspace(0, 1, num_thresholds)
    accuracies = [evaluate_threshold(y_true, y_pred, threshold) for threshold in thresholds]
    print("max sim score:", max(y_pred))
    print("max f1 score: ", max(accuracies))
    best_threshold = thresholds[np.argmax(accuracies)]
    return best_threshold


if __name__ == '__main__':
    hdf5_file_path = "./jfg_vectors_comparison_0_tex_vid.hdf5"
    dataset = jfg_comparison_generator(hdf5_file_path, 500, 128)
    BATCH_SIZE = 100
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    comparator = load_model("./jfg_comparator_s1_tex_vid.h5")

    # get similarity scores for a validation dataset
    sim_scores, y_true_val = generate_sim_scores_and_labels(comparator, dataset)
    print("sim_scores:", sim_scores)
    print("y_true_val:", y_true_val)

    best_threshold = find_best_threshold(y_true_val, sim_scores)
    print("best_threshold: ", best_threshold)

    # Make binary predictions using the best_threshold
    binary_predictions = sim_scores > best_threshold
    print(sum(binary_predictions) / len(binary_predictions), binary_predictions)
