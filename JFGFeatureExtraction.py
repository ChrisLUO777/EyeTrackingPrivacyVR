import itertools

import h5py
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
from keras.models import load_model
from JFGComparator import create_jfg_comparator
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


def binary_crossentropy_loss(y_true, y_pred):
    loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return loss


def extract_jfg_features_step(jfg_model, x_batch, y_batch):
    # Create all pairs of comparison in the batch
    x1, x2, y_true = generate_pairs(x_batch, y_batch)

    # Use the existing model to process both inputs and generate two feature vectors
    f1 = jfg_model(x1, training=False)
    f2 = jfg_model(x2, training=False)

    return f1, f2, y_true


# Function to save a batch of gazemaps of multiple users into a hdf5 file
def save_jfg_vector_to_hdf5(jfg1, jfg2, y_true, file_index,):
    with h5py.File("jfg_vectors_comparison_" + str(file_index) + "_tex_vid.hdf5", "w") as hdf5_file:
        num_comps_to_add = len(y_true)
        vector_shape = (num_comps_to_add, 128)
        y_true_shape = (num_comps_to_add,)
        hdf5_file.create_dataset("jfg1",
                                 shape=vector_shape,
                                 compression="gzip", compression_opts=9)
        hdf5_file.create_dataset("jfg2",
                                 shape=vector_shape,
                                 compression="gzip", compression_opts=9)
        hdf5_file.create_dataset("y_true",
                                 shape=y_true_shape,
                                 compression="gzip", compression_opts=9)

        for vector_index in range(num_comps_to_add):
            hdf5_file["jfg1"][vector_index] = jfg1[vector_index]
            hdf5_file["jfg2"][vector_index] = jfg2[vector_index]
            hdf5_file["y_true"][vector_index] = y_true[vector_index]


if __name__ == '__main__':
    # Usage:
    hdf5_file_path = "./train_jfg_50users_s1_video_s1_r2r3_text.hdf5"
    dataset = gaze_map_generator(hdf5_file_path, 240, 6400)
    # dataset = dataset.take(20)
    BATCH_SIZE = 10
    # Apply preprocessing with multiple workers
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    jfg_model = load_model("./jfg_model_tex_vid.h5")
    jfg_vectors1 = []
    jfg_vectors2 = []
    y_true_labels = []
    file_counter = 0

    for x_batch, y_batch in dataset:
        jfg1, jfg2, y_true = extract_jfg_features_step(jfg_model, x_batch, y_batch)
        jfg_vectors1.extend(jfg1)
        jfg_vectors2.extend(jfg2)
        y_true_labels.extend(y_true)
        if len(y_true_labels) >= 10000:
            save_jfg_vector_to_hdf5(jfg_vectors1, jfg_vectors2, y_true_labels, file_counter)
            jfg_vectors1 = []
            jfg_vectors2 = []
            y_true_labels = []
            file_counter += 1

    if y_true_labels:
        save_jfg_vector_to_hdf5(jfg_vectors1, jfg_vectors2, y_true_labels, file_counter)
