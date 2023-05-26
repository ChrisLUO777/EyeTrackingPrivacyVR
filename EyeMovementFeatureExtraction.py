# This is the main program to segment and extract features from the eye movement data.
import glob
from pathlib import Path
import math
import os
import numpy as np
from scipy import signal
import pandas as pd
import collections

from EyeMovementRecognition import get_smooth_eye_movement_speed, get_optimal_peak_saccade_velocity_threshold, \
    get_saccade_index, get_fixation_index, get_blink_index, get_real_number_eye_gaze_coordinates
from SaccadeFeatureExtraction import get_saccade_duration_distribution, get_saccade_speed_distribution, \
    get_saccade_right_rotation_distribution, get_saccade_left_rotation_distribution, \
    get_saccade_start_time_distribution, get_saccade_up_rotation_distribution, get_saccade_down_rotation_distribution, \
    get_saccade_horizontal_range_distribution, get_saccade_vertical_range_distribution, \
    get_fixation_duration_distribution, get_fixation_start_time_distribution, \
    get_fixation_horizontal_centroid_distribution, get_fixation_vertical_centroid_distribution, \
    get_blink_duration_distribution, get_blink_start_time_distribution


def extract_features(left_eye_gaze_coordinates, right_eye_gaze_coordinates):
    feature_set = []
    # process left eye gaze trace data.
    real_number_left_eye_gaze_coordinates = get_real_number_eye_gaze_coordinates(left_eye_gaze_coordinates)
    smooth_left_eye_speed = get_smooth_eye_movement_speed(left_eye_gaze_coordinates)
    updated_left_peak_saccade_velocity_threshold = get_optimal_peak_saccade_velocity_threshold(smooth_left_eye_speed)
    left_eye_saccade_index = get_saccade_index(smooth_left_eye_speed, updated_left_peak_saccade_velocity_threshold)

    # extract 9 saccade features from left eye gaze trace.
    left_eye_saccade_durations = get_saccade_duration_distribution(left_eye_saccade_index)
    left_eye_saccade_start_time = get_saccade_start_time_distribution(left_eye_saccade_index)
    left_eye_saccade_speed = get_saccade_speed_distribution(smooth_left_eye_speed, left_eye_saccade_index)
    left_eye_saccade_right_rotation = get_saccade_right_rotation_distribution(real_number_left_eye_gaze_coordinates,
                                                                              left_eye_saccade_index)
    left_eye_saccade_left_rotation = get_saccade_left_rotation_distribution(real_number_left_eye_gaze_coordinates,
                                                                            left_eye_saccade_index)
    left_eye_saccade_up_rotation = get_saccade_up_rotation_distribution(real_number_left_eye_gaze_coordinates,
                                                                        left_eye_saccade_index)
    left_eye_saccade_down_rotation = get_saccade_down_rotation_distribution(real_number_left_eye_gaze_coordinates,
                                                                            left_eye_saccade_index)
    left_eye_saccade_horizontal_range = get_saccade_horizontal_range_distribution(real_number_left_eye_gaze_coordinates,
                                                                                  left_eye_saccade_index)
    left_eye_saccade_vertical_range = get_saccade_vertical_range_distribution(real_number_left_eye_gaze_coordinates,
                                                                              left_eye_saccade_index)

    append_features(feature_set, left_eye_saccade_durations)
    append_features(feature_set, left_eye_saccade_start_time)
    append_features(feature_set, left_eye_saccade_speed)
    append_features(feature_set, left_eye_saccade_right_rotation)
    append_features(feature_set, left_eye_saccade_left_rotation)
    append_features(feature_set, left_eye_saccade_up_rotation)
    append_features(feature_set, left_eye_saccade_down_rotation)
    append_features(feature_set, left_eye_saccade_horizontal_range)
    append_features(feature_set, left_eye_saccade_vertical_range)

    left_eye_fixation_index = get_fixation_index(smooth_left_eye_speed, updated_left_peak_saccade_velocity_threshold)

    # extract 4 fixation features from left eye gaze trace.
    left_eye_fixation_duration = get_fixation_duration_distribution(left_eye_fixation_index)
    left_eye_fixation_start_time = get_fixation_start_time_distribution(left_eye_fixation_index)
    left_eye_fixation_horizontal_centroid = get_fixation_horizontal_centroid_distribution(
        real_number_left_eye_gaze_coordinates, left_eye_fixation_index)
    left_eye_fixation_vertical_centroid = get_fixation_vertical_centroid_distribution(
        real_number_left_eye_gaze_coordinates, left_eye_fixation_index)

    append_features(feature_set, left_eye_fixation_duration)
    append_features(feature_set, left_eye_fixation_start_time)
    append_features(feature_set, left_eye_fixation_horizontal_centroid)
    append_features(feature_set, left_eye_fixation_vertical_centroid)

    left_eye_blink_index = get_blink_index(left_eye_gaze_coordinates)

    # extract 2 blink features from left eye gaze trace
    left_eye_blink_duration = get_blink_duration_distribution(left_eye_blink_index)
    left_eye_blink_start_time = get_blink_start_time_distribution(left_eye_blink_index)

    append_features(feature_set, left_eye_blink_duration)
    append_features(feature_set, left_eye_blink_start_time)

    # process right eye gaze trace data.
    real_number_right_eye_gaze_coordinates = get_real_number_eye_gaze_coordinates(right_eye_gaze_coordinates)
    smooth_right_eye_speed = get_smooth_eye_movement_speed(right_eye_gaze_coordinates)
    updated_right_peak_saccade_velocity_threshold = get_optimal_peak_saccade_velocity_threshold(smooth_right_eye_speed)
    right_eye_saccade_index = get_saccade_index(smooth_right_eye_speed, updated_right_peak_saccade_velocity_threshold)

    # extract 9 saccade features from right eye gaze trace.
    right_eye_saccade_durations = get_saccade_duration_distribution(right_eye_saccade_index)
    right_eye_saccade_start_time = get_saccade_start_time_distribution(right_eye_saccade_index)
    right_eye_saccade_speed = get_saccade_speed_distribution(smooth_right_eye_speed, right_eye_saccade_index)
    right_eye_saccade_right_rotation = get_saccade_right_rotation_distribution(real_number_right_eye_gaze_coordinates,
                                                                               right_eye_saccade_index)
    right_eye_saccade_left_rotation = get_saccade_left_rotation_distribution(real_number_right_eye_gaze_coordinates,
                                                                             right_eye_saccade_index)
    right_eye_saccade_up_rotation = get_saccade_up_rotation_distribution(
        real_number_right_eye_gaze_coordinates, right_eye_saccade_index)
    right_eye_saccade_down_rotation = get_saccade_down_rotation_distribution(real_number_right_eye_gaze_coordinates,
                                                                             right_eye_saccade_index)
    right_eye_saccade_horizontal_range = get_saccade_horizontal_range_distribution(
        real_number_right_eye_gaze_coordinates, right_eye_saccade_index)
    right_eye_saccade_vertical_range = get_saccade_vertical_range_distribution(real_number_right_eye_gaze_coordinates,
                                                                               right_eye_saccade_index)

    append_features(feature_set, right_eye_saccade_durations)
    append_features(feature_set, right_eye_saccade_start_time)
    append_features(feature_set, right_eye_saccade_speed)
    append_features(feature_set, right_eye_saccade_right_rotation)
    append_features(feature_set, right_eye_saccade_left_rotation)
    append_features(feature_set, right_eye_saccade_up_rotation)
    append_features(feature_set, right_eye_saccade_down_rotation)
    append_features(feature_set, right_eye_saccade_horizontal_range)
    append_features(feature_set, right_eye_saccade_vertical_range)

    right_eye_fixation_index = get_fixation_index(smooth_right_eye_speed, updated_right_peak_saccade_velocity_threshold)

    # extract 4 fixation features from right eye trace.
    right_eye_fixation_duration = get_fixation_duration_distribution(right_eye_fixation_index)
    right_eye_fixation_start_time = get_fixation_start_time_distribution(right_eye_fixation_index)
    right_eye_fixation_horizontal_centroid = get_fixation_horizontal_centroid_distribution(
        real_number_right_eye_gaze_coordinates, right_eye_fixation_index)
    right_eye_fixation_vertical_centroid = get_fixation_vertical_centroid_distribution(
        real_number_right_eye_gaze_coordinates, right_eye_fixation_index)

    append_features(feature_set, right_eye_fixation_duration)
    append_features(feature_set, right_eye_fixation_start_time)
    append_features(feature_set, right_eye_fixation_horizontal_centroid)
    append_features(feature_set, right_eye_fixation_vertical_centroid)

    right_eye_blink_index = get_blink_index(right_eye_gaze_coordinates)

    # extract 2 blink features from right eye trace.
    right_eye_blink_duration = get_blink_duration_distribution(right_eye_blink_index)
    right_eye_blink_start_time = get_blink_start_time_distribution(right_eye_blink_index)

    append_features(feature_set, right_eye_blink_duration)
    append_features(feature_set, right_eye_blink_start_time)

    return feature_set


def append_features(whole_features, new_feature):
    # append new features (2 rows) into the whole feature set.
    whole_features.append([val for val in new_feature[0]])
    whole_features.append([val for val in new_feature[1]])


if __name__ == '__main__':
    # path = './gazebasevr/data/'
    # extension = 'csv'
    # os.chdir(path)  # note this command change current directory into [path].
    # eye_coordinate_files = glob.glob('*.{}'.format(extension))
    # for eye_coordinate_file in eye_coordinate_files:
    #     eye_coordinate_file_name = eye_coordinate_file.split('.')[0]
    #     print(eye_coordinate_file_name.split('_'))
    #     activity_name = eye_coordinate_file_name.split('_')[-1]
    #     if activity_name == "VID":
    #         user_id = eye_coordinate_file_name.split('_')[1][1:]
    #         print(user_id)
    #         df = pd.read_csv(eye_coordinate_file)

    path = './gazebasevr/data/'
    os.chdir(path)  # note this command change current directory into [path].
    extension = 'csv'
    eye_coordinate_files = glob.glob('*.{}'.format(extension))
    user_id2VID_record_num = collections.defaultdict(int)
    user_id2TEX_record_num = collections.defaultdict(int)
    for eye_coordinate_file in eye_coordinate_files:
        record_length = 8 * 250     # the eye movement record is divided into 8s segments.
        eye_coordinate_file_name = eye_coordinate_file.split('.')[0]
        print(eye_coordinate_file_name.split('_'))
        activity_name = eye_coordinate_file_name.split('_')[-1]
        if activity_name == "VID":
            df = pd.read_csv(eye_coordinate_file)
            user_id = eye_coordinate_file_name.split('_')[1][1:]
            print(user_id + ", VID, " + str(user_id2VID_record_num[user_id]))
            # derive left and right eye gaze trace data.
            left_eye_gaze_coordinates_data = [coordinate for coordinate in list(zip(df.get("lx"), df.get("ly")))]
            right_eye_gaze_coordinates_data = [coordinate for coordinate in list(zip(df.get("rx"), df.get("ry")))]

            record_start_index = 0
            for record_end_index in range(len(left_eye_gaze_coordinates_data)):
                if record_end_index - record_start_index == record_length:
                    # extract 30 features from left and right eye gaze data respectively.
                    features = extract_features(left_eye_gaze_coordinates_data[record_start_index:record_end_index + 1],
                                                right_eye_gaze_coordinates_data[record_start_index:record_end_index + 1]
                                                )
                    record_start_index = record_end_index

                    features_np = np.array(features)
                    df_save = pd.DataFrame(features_np)
                    df_save.to_csv("./8s_feature/VID/" + user_id + '_' + str(user_id2VID_record_num[user_id]) + ".csv",
                                   index=False, header=False)
                    user_id2VID_record_num[user_id] += 1
        elif activity_name == "TEX":
            df = pd.read_csv(eye_coordinate_file)
            user_id = eye_coordinate_file_name.split('_')[1][1:]
            print(user_id + ", TEX, " + str(user_id2TEX_record_num[user_id]))
            # derive left and right eye gaze trace data.
            left_eye_gaze_coordinates_data = [coordinate for coordinate in list(zip(df.get("lx"), df.get("ly")))]
            right_eye_gaze_coordinates_data = [coordinate for coordinate in list(zip(df.get("rx"), df.get("ry")))]

            record_start_index = 0
            for record_end_index in range(len(left_eye_gaze_coordinates_data)):
                if record_end_index - record_start_index == record_length:

                    # extract 30 features from left and right eye gaze data respectively.
                    features = extract_features(left_eye_gaze_coordinates_data[record_start_index:record_end_index + 1],
                                                right_eye_gaze_coordinates_data[record_start_index:record_end_index + 1]
                                                )
                    record_start_index = record_end_index

                    features_np = np.array(features)
                    df_save = pd.DataFrame(features_np)
                    df_save.to_csv("./8s_feature/TEX/" + user_id + '_' + str(user_id2TEX_record_num[user_id]) + ".csv",
                                   index=False, header=False)
                    user_id2TEX_record_num[user_id] += 1
