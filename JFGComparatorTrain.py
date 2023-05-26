import itertools

import h5py
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
from keras.models import load_model
from JFGComparator import create_jfg_comparator
from JointFeatureGeneratorTrain import gaze_map_generator


def jfg_comparison_generator(file, height, width):
    def _generator():
        with h5py.File(file, 'r') as f:
            jfg1 = f['jfg1']
            jfg2 = f['jfg2']
            y_true = f['y_true']

            for comp_index in range(len(y_true)):
                yield jfg1[comp_index], jfg2[comp_index], y_true[comp_index]

    # data types of the user_id and respective gaze_map matrix
    output_types = (tf.float32, tf.float32, tf.int32)
    # Specify the output shapes of the generator function
    output_shapes = ((width,), (width,), ())

    return tf.data.Dataset.from_generator(_generator,
                                          output_types=output_types,
                                          output_shapes=output_shapes)


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


def train_step(model, f1, f2, y_true, optimizer):
    # print("f1, f2:", f1, f2)
    with tf.GradientTape() as tape:
        # Compute the similarity score using the target model
        y_pred = model([f1, f2], training=True)

        # Compute binary cross-entropy loss
        loss = binary_crossentropy_loss(y_true, y_pred)

    # Update the model's parameters
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss.numpy()


if __name__ == '__main__':
    # Usage:
    # hdf5_file_path = "./train_jfg_50users_s1_vid_8s.hdf5"
    # dataset = gaze_map_generator(hdf5_file_path, 240, 6400)
    # dataset = dataset.take(20)
    # BATCH_SIZE = 10
    # # Apply preprocessing with multiple workers
    # dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    hdf5_file_path = "./jfg_vectors_comparison_0_tex_vid.hdf5"
    dataset = jfg_comparison_generator(hdf5_file_path, 500, 128)
    BATCH_SIZE = 100
    dataset = dataset.shuffle(buffer_size=100).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    comparator = create_jfg_comparator(128)

    num_epochs = 10
    optimizer = Adam(learning_rate=0.001)
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for jfg1, jfg2, y_batch in dataset:
            loss = train_step(comparator, jfg1, jfg2, y_batch, optimizer)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

    comparator.save("./jfg_comparator_s1_tex_vid.h5")
