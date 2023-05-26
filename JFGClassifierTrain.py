import random

import tensorflow as tf
import numpy as np
import glob
import h5py

# Function to load a hdf5 file as the training data.
from JointFeatureGenerator import create_joint_feature_generator, create_jfg_classifier
from JointFeatureGeneratorTrain import gaze_map_generator, parallel_hdf5_to_dataset


def extract_user_id(gaze_map, user_id):
    # Function to count the number of classes in a tf.Data.dataset
    # Assuming the record is a tuple, where the first element is the data,
    # and the second element is the user_id
    return user_id


def generate_one_hot_encoded_labels(sample, label, min_label, num_classes):
    one_hot_label = tf.one_hot(label - min_label, num_classes)
    return sample, one_hot_label


if __name__ == '__main__':
    hdf5_file_path = "./train_jfg_50users_s1_video_s1_r2r3_text.hdf5"
    dataset = parallel_hdf5_to_dataset(hdf5_file_path)
    # train_dataset = dataset.take(20)
    train_dataset = dataset
    num_unique_users = 50

    # hdf5_file_paths = ['./train_jfg_50users_s1_video_with_labels.hdf5', './train_jfg_50users_s1_text_with_labels.hdf5']
    # # Load and process data from multiple HDF5 files
    # all_datasets = [
    #     parallel_hdf5_to_dataset(file_path)
    #     for file_path in hdf5_file_paths
    # ]
    # train_dataset = tf.data.Dataset.concatenate(*all_datasets)

    # Train the classifier model
    # Count the number of classes.
    # unique_user_ids = set()
    # for user_id in train_dataset.map(extract_user_id):
    #     unique_user_ids.add(user_id.numpy())
    # num_unique_users = len(unique_user_ids)
    # min_user_id = min(unique_user_ids)
    # print(f'Number of unique user_ids: {num_unique_users}')
    # train_dataset = train_dataset.map(lambda sample, label: generate_one_hot_encoded_labels(sample, label, min_user_id,
    #                                                                                         num_unique_users),
    #                                   num_parallel_calls=tf.data.AUTOTUNE)

    BATCH_SIZE = 30
    train_dataset = train_dataset.shuffle(buffer_size=100).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    eme_input_shape = (240, 6400)
    jfg_model = create_joint_feature_generator(eme_input_shape)
    jfg_classifier = create_jfg_classifier(jfg_model, num_unique_users)
    jfg_classifier.fit(train_dataset, epochs=100, workers=8, use_multiprocessing=True)
    jfg_model.save("./jfg_model_c_tex_vid.h5")
    jfg_classifier.save("./jfg_classifier_tex_vid.h5")
