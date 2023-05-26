import glob
import os
import math
import numpy as np
import pandas as pd
import collections
import cv2
import h5py


def sphere_coordinate_to_equi_coordinate(sphere_x, sphere_y, frame_width, frame_height):
    # the origin point of x starts at the middle of a row.
    equi_x = int(sphere_x * frame_width / 360) + frame_width // 2
    if equi_x >= frame_width:
        equi_x -= frame_width
    # the origin point of y starts at the top of a column.
    equi_y = int((1 - math.sin(sphere_y)) * frame_height / 2)
    return [equi_x, equi_y]


# generate one gaze map from one eye movement record
# (also concatenate each sample with the respective vce feature vector).
def generate_flatten_equi_gaze_map(left_eye_gaze_coordinates, right_eye_gaze_coordinates, cyclope_eye_gaze_coordinates,
                                   equi_frame_width, equi_frame_height, vce_features):
    equi_coordinates = []
    batch_index = 0
    # use a batch to average multiple gaze maps into one to match with each vce_feature vector
    # note: batch_size may be a float value
    batch_size = min(len(left_eye_gaze_coordinates), 10000) / len(vce_features)
    # down sample the 250 hz (gaze map) to 24 hz (video frame)
    while batch_index < len(vce_features):
        equi_frame = np.zeros((equi_frame_height + 1, equi_frame_width + 1), dtype='float32')
        #   average the gaze maps in a batch
        for coordinate_index in range(int(batch_index * batch_size), int((batch_index + 1) * batch_size)):
            if not math.isnan(left_eye_gaze_coordinates[coordinate_index][0]):
                left_equi_x, left_equi_y = sphere_coordinate_to_equi_coordinate(
                    left_eye_gaze_coordinates[coordinate_index][0],
                    left_eye_gaze_coordinates[coordinate_index][1],
                    equi_frame_width, equi_frame_height)
                equi_frame[left_equi_y][left_equi_x] = 1

            if not math.isnan(right_eye_gaze_coordinates[coordinate_index][0]):
                right_equi_x, right_equi_y = sphere_coordinate_to_equi_coordinate(
                    right_eye_gaze_coordinates[coordinate_index][0],
                    right_eye_gaze_coordinates[coordinate_index][1],
                    equi_frame_width, equi_frame_height)
                equi_frame[right_equi_y][right_equi_x] = 1

            if not math.isnan(cyclope_eye_gaze_coordinates[coordinate_index][0]):
                cyclope_equi_x, cyclope_equi_y = sphere_coordinate_to_equi_coordinate(
                    cyclope_eye_gaze_coordinates[coordinate_index][0],
                    cyclope_eye_gaze_coordinates[coordinate_index][1],
                    equi_frame_width, equi_frame_height)
                equi_frame[cyclope_equi_y][cyclope_equi_x] = 1

        # Apply Gaussian blur to the 2D matrix
        kernel_size = (15, 15)  # Kernel size should be an odd number
        blurred_equi_frame = cv2.GaussianBlur(equi_frame, kernel_size, 3)
        flatten_blurred_equi_frame = blurred_equi_frame.flatten()
        # each vce_features[batch_index] is a nested array (e.g., [[1, 2, 3...]]).
        concatenate_vce_gaze_vector = np.concatenate((vce_features[batch_index][0], flatten_blurred_equi_frame))
        concatenate_vce_gaze_vector_padded = np.pad(concatenate_vce_gaze_vector,
                                                    (0, 6400 - len(concatenate_vce_gaze_vector)), mode='constant')
        equi_coordinates.append(concatenate_vce_gaze_vector_padded)

        batch_index += 1

    print("length of vce_features[0][0]:", len(vce_features[0][0]))
    # pad the gaze map to 960 samples
    for _ in range(960 - len(equi_coordinates)):
        empty_flatten_frame = np.zeros(6400, dtype='float32')
        equi_coordinates.append(empty_flatten_frame)
    print("shape of a gaze map:", len(equi_coordinates), len(equi_coordinates[0]))
    return equi_coordinates[:960]


# Function to save a batch of gazemaps of multiple users into a hdf5 file
def save_batch_to_hdf5(batch_gaze_maps, batch_user_id_labels, batch_masks, max_length, height, width, file_index):
    """
    :param batch_gaze_maps: a list of 3D matrix. Each 3D matrix is a list of gaze maps of a user.
    :param batch_user_id_labels: a list of int recording the user_id.
    :param batch_masks: a list of mask arrays recording which gaze maps are padded empty matrix.
    :param max_length: the max number of gaze maps a user can have
    :param height: the length of a gaze map (set as 960 later)
    :param width: the size of a flattened gaze map, i.e., (frame_height + 1) * (frame_width + 1)
    :param file_index: the counter to mark how many files have been saved.
    :return:
    """
    with h5py.File("gaze_maps_compressed_" + str(file_index) + ".hdf5", "w") as hdf5_file:
        num_users_to_add = len(batch_gaze_maps)
        gazemap_shape = (num_users_to_add, max_length, height, width)
        hdf5_file.create_dataset("gaze_maps",
                                 shape=gazemap_shape,
                                 compression="gzip", compression_opts=9)
        hdf5_file.create_dataset("user_id", shape=(num_users_to_add, max_length),
                                 compression="gzip", compression_opts=9)
        # create a binary padding mask to record which samples are the padded ones.
        hdf5_file.create_dataset("masks", shape=(num_users_to_add, max_length),
                                 compression="gzip", compression_opts=9)

        for user_index in range(num_users_to_add):
            hdf5_file["gaze_maps"][user_index] = batch_gaze_maps[user_index]
            hdf5_file["user_id"][user_index] = batch_user_id_labels[user_index]
            hdf5_file["masks"][user_index] = batch_masks[user_index]


if __name__ == '__main__':
    equi_width = 64
    equi_height = 64

    path = './gazebasevr/data/'
    os.chdir(path)  # note this command change current directory into [path].
    extension = 'csv'
    eye_coordinate_files = glob.glob('*.{}'.format(extension))
    user_id2eye_coordinate_file = collections.defaultdict(list)
    for eye_coordinate_file in eye_coordinate_files:
        eye_coordinate_file_name = eye_coordinate_file.split('.')[0]
        activity_name = eye_coordinate_file_name.split('_')[-1]
        round_num = eye_coordinate_file_name.split('_')[1][0]
        user_id = eye_coordinate_file_name.split('_')[1][1:]
        session_name = eye_coordinate_file_name.split('_')[2]
        # if (user_id == "038" or user_id == "039") and activity_name == "VID" and session_name == "S1":  # test with a smaller dataset
        # if activity_name == "VID" and session_name == "S1":     # extracts eye movement data from video session 1
        # if activity_name == "TEX" and session_name == "S1" and (round_num == "2" or round_num == "3"):  # r2/r3 s1 tex
        if (activity_name == "TEX" and session_name == "S1" and (round_num == "2" or round_num == "3")) or \
                (activity_name == "VID" and session_name == "S1"):
            user_id2eye_coordinate_file[user_id].append(eye_coordinate_file)
            print(eye_coordinate_file_name.split('_'))
            print(user_id + ", VID or TEX.")

    # Save each batch of user's data into a compressed hdf5 file.
    max_num_gaze_maps_per_user = 48
    num_users = len(user_id2eye_coordinate_file.keys())
    batch_size = 50
    batch_gaze_maps_of_users = []
    batch_user_id_labels = []
    batch_masks = []
    file_index = 0
    # vce_features = np.load('video1_feature.npy')
    vce_features_for_vid = np.load('video1_feature.npy')
    vce_features_for_tex = np.load('text1_feature.npy')

    for user_id in user_id2eye_coordinate_file.keys():
        gaze_maps = []
        for eye_coordinate_file in user_id2eye_coordinate_file[user_id]:
            # get the activity name, which decides the vce feature vector to be concatenated.
            eye_coordinate_file_name = eye_coordinate_file.split('.')[0]
            activity_name = eye_coordinate_file_name.split('_')[-1]

            df = pd.read_csv(eye_coordinate_file)
            print(eye_coordinate_file)
            left_eye_coordinates = [coordinate for coordinate in list(zip(df.get("lx"), df.get("ly")))]
            right_eye_coordinates = [coordinate for coordinate in list(zip(df.get("rx"), df.get("ry")))]
            cyclope_eye_coordinates = [coordinate for coordinate in list(zip(df.get("x"), df.get("y")))]
            if activity_name == "TEX":
                equi_coordinates = generate_flatten_equi_gaze_map(left_eye_coordinates, right_eye_coordinates,
                                                                  cyclope_eye_coordinates,
                                                                  equi_width, equi_height, vce_features_for_tex)
            else:
                equi_coordinates = generate_flatten_equi_gaze_map(left_eye_coordinates, right_eye_coordinates,
                                                                  cyclope_eye_coordinates,
                                                                  equi_width, equi_height, vce_features_for_vid)
            print("shape of equi_coordinates: ", len(equi_coordinates), len(equi_coordinates[0]))
            equi_coordinates_np = np.vstack(equi_coordinates)
            equi_coordinates_np_8s = np.array_split(equi_coordinates_np, 4)  # split the gaze map into 4 8s-segments
            gaze_maps.extend(equi_coordinates_np_8s)
            print("shape of equi_coordinates_np: ", equi_coordinates_np.shape)

        num_gaze_maps = len(gaze_maps)
        print("number of gaze maps for this user: ", num_gaze_maps)
        print("shape of the first gaze map for this user:", gaze_maps[0].shape)
        padding_length = max_num_gaze_maps_per_user - num_gaze_maps
        # Pad empty matrix so each user has [max_num_gaze_maps_per_user] number of gaze maps.
        gaze_maps_of_a_user_padded = np.pad(gaze_maps, ((0, padding_length), (0, 0), (0, 0)), mode='constant')

        # use an array to store the user ids
        user_id_labels = np.zeros(max_num_gaze_maps_per_user)
        user_id_labels[:] = user_id

        # Use a mask array to record which gaze maps are the padded empty matrix.
        mask = np.zeros(max_num_gaze_maps_per_user)
        mask[:num_gaze_maps] = 1

        batch_gaze_maps_of_users.append(gaze_maps_of_a_user_padded)
        batch_user_id_labels.append(user_id_labels)
        batch_masks.append(mask)

        # If the batch collected data of enough users, save them to a hdf5 file.
        if len(batch_gaze_maps_of_users) == batch_size:
            save_batch_to_hdf5(batch_gaze_maps_of_users, batch_user_id_labels, batch_masks, max_num_gaze_maps_per_user,
                               240, 6400, file_index)
            batch_gaze_maps_of_users = []
            batch_user_id_labels = []
            batch_masks = []
            file_index += 1

        if file_index == 4:
            break

    # If there are left gaze_maps, save them to a new hdf5 file.
    if batch_gaze_maps_of_users:
        save_batch_to_hdf5(batch_gaze_maps_of_users, batch_user_id_labels, batch_masks, max_num_gaze_maps_per_user,
                           240, 6400, file_index)
        file_index += 1
