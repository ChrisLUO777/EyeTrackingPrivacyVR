from keras.models import load_model
import tensorflow as tf
from JointFeatureGeneratorTrain import gaze_map_generator
import numpy as np


def extract_user_id(gaze_map, user_id):
    # Function to count the number of classes in a tf.Data.dataset
    # Assuming the record is a tuple, where the first element is the data,
    # and the second element is the user_id
    return user_id


def generate_one_hot_encoded_labels(sample, label, min_label, num_classes):
    one_hot_label = tf.one_hot(label - min_label, num_classes)
    return sample, one_hot_label


if __name__ == '__main__':
    hdf5_file_path = "./train_jfg_50users_s1_vid_8s.hdf5"
    dataset = gaze_map_generator(hdf5_file_path, 240, 6400)
    dataset = dataset.take(20)

    # Count the number of classes.
    unique_user_ids = set()
    for user_id in dataset.map(extract_user_id):
        unique_user_ids.add(user_id.numpy())
    num_unique_users = len(unique_user_ids)
    min_user_id = min(unique_user_ids)
    dataset = dataset.map(lambda sample, label: generate_one_hot_encoded_labels(sample, label, min_user_id,
                                                                                num_unique_users),
                          num_parallel_calls=tf.data.AUTOTUNE)

    BATCH_SIZE = 10
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    jfg_classifier = load_model("./jfg_classifier.h5")
    jfg_classifier.evaluate(dataset)
    for x_batch, y_batch in dataset:
        y_pred = jfg_classifier(x_batch, training=False)
        print("y_batch: ", y_batch)
        print("jfg_classifier y_pred argmax: ", np.argmax(y_pred, axis=1))

    jfg_model = load_model("./jfg_model_c.h5")
    for x_batch, y_batch in dataset:
        y_pred = jfg_model(x_batch, training=False)
        print("jfg_model y_pred: ", y_pred)
