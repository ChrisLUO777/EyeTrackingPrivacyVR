from keras.models import load_model
import tensorflow as tf
from JointFeatureGeneratorTrain import gaze_map_generator
import numpy as np


def evaluate_triplet_loss(model, dataset, margin):
    total_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in dataset:
        y_pred = model(x_batch, training=False)
        print("y_pred: ", np.mean(y_pred, axis=1))
        print("y_batch: ", y_batch)

        # Compute distances between all pairs of embeddings
        dist = tf.reduce_sum(tf.square(y_pred[:, tf.newaxis] - y_pred), axis=-1)

        # Compute anchor-positive and anchor-negative distances
        same_identity_mask = tf.equal(y_batch[:, tf.newaxis], y_batch)
        positive_dist = tf.expand_dims(tf.linalg.diag_part(dist), axis=1)
        anchor_positive_dist = tf.where(same_identity_mask, dist, positive_dist)

        negative_dist = tf.expand_dims(tf.linalg.diag_part(dist), axis=0)
        anchor_negative_dist = tf.where(same_identity_mask, negative_dist, dist)

        # Compute triplet loss
        # loss = tf.maximum(anchor_positive_dist - anchor_negative_dist + margin, 0.0)
        loss = anchor_positive_dist - anchor_negative_dist
        mean_loss = tf.reduce_mean(loss)
        print("dist: ", dist)

        total_loss += mean_loss.numpy()
        num_batches += 1

    # Calculate the average loss over all batches
    avg_loss = total_loss / num_batches
    return avg_loss


if __name__ == '__main__':
    max_num_gaze_maps_per_user = 12
    hdf5_file_path = "./train_jfg_50users_s1_video_s1_r2r3_text.hdf5"
    dataset = gaze_map_generator(hdf5_file_path, 240, 6400)
    # dataset = dataset.take(20)
    BATCH_SIZE = 10
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    jfg_model = load_model("./jfg_model_tex_vid.h5")

    # evaluate the jfg model using the triplet loss.
    margin = 1.0  # Define the margin value used during training
    eval_loss = evaluate_triplet_loss(jfg_model, dataset, margin)
    print(f'Triplet loss on evaluation dataset: {eval_loss}')
