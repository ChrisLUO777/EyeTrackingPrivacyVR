import random
from keras.models import load_model
import tensorflow as tf
import numpy as np
import glob
import h5py
from multiprocessing import cpu_count

# Function to load a hdf5 file as the training data.
from JointFeatureGenerator import create_joint_feature_generator, create_jfg_classifier


def gaze_map_generator(file, height, width):
    def _generator():
        with h5py.File(file, 'r') as f:
            gaze_maps = f['gaze_maps']
            # user_id_labels = f['user_id']
            masks = f['masks']

            for user_index, user_gaze_maps in enumerate(gaze_maps):
                # user_id = user_id_labels[user_index]
                user_masks = masks[user_index]
                for gaze_map_idx, is_not_empty in enumerate(user_masks):
                    if is_not_empty:
                        yield user_gaze_maps[gaze_map_idx], user_index
                        # yield user_gaze_maps[gaze_map_idx], int(user_id[gaze_map_idx])

    # data types of the user_id and respective gaze_map matrix
    output_types = (tf.float32, tf.int32)
    # Specify the output shapes of the generator function
    output_shapes = ((height, width), ())

    return tf.data.Dataset.from_generator(_generator,
                                          output_types=output_types,
                                          output_shapes=output_shapes)


def parallel_hdf5_to_dataset(file_path, num_parallel_calls=None):
    if num_parallel_calls is None:
        num_parallel_calls = cpu_count()

    # Convert numpy arrays to tf.data.Dataset
    dataset = gaze_map_generator(file_path, 240, 6400)

    # Parallelize the loading and preprocessing using the num_parallel_calls parameter
    dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=num_parallel_calls)

    return dataset


def extract_user_id(gaze_map, user_id):
    # Function to count the number of classes in a tf.Data.dataset
    # Assuming the record is a tuple, where the first element is the data,
    # and the second element is the user_id
    return user_id


def train_step(model, optimizer, x, y, margin=1.0):
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = model(x, training=True)

        # Compute pairwise Euclidean distance between all elements in y_pred and store them in a matrix
        dist = tf.reduce_sum(tf.square(y_pred[:, tf.newaxis] - y_pred), axis=-1)  # size: (batch_size, batch_size)

        same_identity_mask = tf.equal(y[:, tf.newaxis], y)

        # positive_dist is a (batch_size, 1) tensor storing the distance between each sample and itself (approx. 0).
        positive_dist = tf.expand_dims(tf.linalg.diag_part(dist), axis=1)
        # anchor_positive_dist is a (batch_size, batch_size) tensor storing the distance between anchor and positive
        # samples. If sample i and j are from the same user, their distance is as derived in dist. Otherwise, it's
        # replaced with a value approx. 0.
        anchor_positive_dist = tf.where(same_identity_mask, dist, positive_dist)

        # negative_dist is a (1, batch_size) tensor storing the distance between each sample and itself (approx. 0).
        negative_dist = tf.expand_dims(tf.linalg.diag_part(dist), axis=0)
        # anchor_negative_dist is a (batch_size, batch_size) tensor storing the distance between anchor and negative
        # samples. If sample i and j are from the same user, their distance is replaced with a value approx. 0.
        # Otherwise, it's as derived in dist.
        anchor_negative_dist = tf.where(same_identity_mask, negative_dist, dist)

        # Easy triplets with big difference in distance achieves a loss of 0.
        # loss = tf.maximum(anchor_positive_dist - anchor_negative_dist + margin, 0.0)
        # loss = tf.reduce_mean(loss)

        # # Get the hardest positive and negative distances
        # hardest_positive_dist = tf.reduce_max(positive_dist, axis=1)
        # hardest_negative_dist = tf.reduce_min(negative_dist, axis=1)
        #
        # # Calculate the triplet loss for the hardest triplets
        # loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
        #
        # # Get the top k% hardest triplets
        # hard_triplet_ratio = 0.3
        # k = int(hard_triplet_ratio * tf.cast(tf.shape(loss)[0], tf.float32))
        # top_k_losses, _ = tf.nn.top_k(loss, k=k)
        #
        # # Calculate the mean loss for the hardest triplets
        # loss = tf.reduce_mean(top_k_losses)
        # print("dist: ", dist)

        loss = tf.maximum(anchor_positive_dist - anchor_negative_dist + margin, 0.0)
        # Get the top k% hardest triplets
        hard_triplet_ratio = 0.3
        k = int(hard_triplet_ratio * tf.cast(tf.shape(loss)[0] * tf.shape(loss)[1], tf.float32))
        top_k_losses, _ = tf.nn.top_k(tf.reshape(loss, (-1,)), k=k)
        loss = tf.reduce_mean(top_k_losses)

    # Compute gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


if __name__ == '__main__':
    max_num_gaze_maps_per_user = 12
    equi_width = 64
    equi_height = 64
    vce_feature_len = 2048

    hdf5_file_path = "./train_jfg_50users_s1_video_s1_r2r3_text.hdf5"

    dataset = gaze_map_generator(hdf5_file_path, 240, 6400)
    # train_dataset = dataset.take(20)
    train_dataset = dataset

    # Train the model using the customized triplet loss function
    epochs = 1
    BATCH_SIZE = 10
    eme_input_shape = (240, 6400)
    # model = create_joint_feature_generator(eme_input_shape)
    model = load_model("./jfg_model_c_tex_vid.h5")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # train_dataset = train_dataset.shuffle(buffer_size=50)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, (x_batch, y_batch) in enumerate(train_dataset.batch(BATCH_SIZE)):
            loss = train_step(model, optimizer, x_batch, y_batch)
            print(f"Step {step + 1}, loss: {loss:.4f}")
    model.save("./jfg_model_tex_vid.h5")

    # # a way to inspect the first five training samples in the dataset
    # for data in dataset.take(5):
    #     user_index, gaze_map = data
    #     print("User index:", user_index.numpy())
    #     print("Gaze map:", gaze_map.numpy())
    #     print("\n")

# # read the hdf5 dataset
# with h5py.File('./gazebasevr/data/gaze_maps_compressed_0.hdf5', 'r') as f:
#     gaze_maps = f['gaze_maps']
#     for user_index in range(len(gaze_maps)):
#         gaze_maps_per_user = gaze_maps[user_index, :, :, :]
#         print(f"{user_index}: gaze maps shape: {gaze_maps_per_user.shape}")
#     pad_masks = f['masks']
#     for user_index in range(len(pad_masks)):
#         pad_masks_per_user = pad_masks[user_index]
#         print(pad_masks_per_user)
