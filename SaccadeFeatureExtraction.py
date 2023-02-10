import math
import numpy as np
from scipy import signal
import pandas as pd


def get_saccade_duration_distribution(saccade_index):
    saccade_durations = []
    for saccade in saccade_index:
        saccade_durations.append(saccade[1] - saccade[0])
    num_bins = 100
    counts, bin_edges = np.histogram(saccade_durations, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_saccade_start_time_distribution(saccade_index):
    saccade_start_time = []
    for saccade in saccade_index:
        saccade_start_time.append(saccade[0])
    num_bins = 100
    counts, bin_edges = np.histogram(saccade_start_time, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_saccade_speed_distribution(smooth_eye_speed, saccade_index):
    saccade_speed = []
    for saccade in saccade_index:
        saccade_speed.append(sum(smooth_eye_speed[saccade[0]:saccade[1]]) / (saccade[1] - saccade[0]))
    num_bins = 100
    counts, bin_edges = np.histogram(saccade_speed, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_saccade_right_rotation_distribution(real_number_eye_gaze_coordinates, saccade_index):
    saccade_right_rotation = []
    for saccade in saccade_index:
        saccade_right_rotation.append(
            max([coordinate[0] for coordinate in real_number_eye_gaze_coordinates[saccade[0]:saccade[1]]]))
    num_bins = 100
    counts, bin_edges = np.histogram(saccade_right_rotation, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_saccade_left_rotation_distribution(real_number_eye_gaze_coordinates, saccade_index):
    saccade_left_rotation = []
    for saccade in saccade_index:
        saccade_left_rotation.append(
            min([coordinate[0] for coordinate in real_number_eye_gaze_coordinates[saccade[0]:saccade[1]]]))
    num_bins = 100
    counts, bin_edges = np.histogram(saccade_left_rotation, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_saccade_up_rotation_distribution(real_number_eye_gaze_coordinates, saccade_index):
    saccade_up_rotation = []
    for saccade in saccade_index:
        saccade_up_rotation.append(
            max([coordinate[1] for coordinate in real_number_eye_gaze_coordinates[saccade[0]:saccade[1]]]))
    num_bins = 100
    counts, bin_edges = np.histogram(saccade_up_rotation, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_saccade_down_rotation_distribution(real_number_eye_gaze_coordinates, saccade_index):
    saccade_down_rotation = []
    for saccade in saccade_index:
        saccade_down_rotation.append(
            min([coordinate[1] for coordinate in real_number_eye_gaze_coordinates[saccade[0]:saccade[1]]]))
    num_bins = 100
    counts, bin_edges = np.histogram(saccade_down_rotation, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_saccade_horizontal_range_distribution(real_number_eye_gaze_coordinates, saccade_index):
    saccade_horizontal_range = []
    for saccade in saccade_index:
        saccade_horizontal_range.append(
            max([coordinate[0] for coordinate in real_number_eye_gaze_coordinates[saccade[0]:saccade[1]]]) -
            min([coordinate[0] for coordinate in real_number_eye_gaze_coordinates[saccade[0]:saccade[1]]]))
    num_bins = 100
    counts, bin_edges = np.histogram(saccade_horizontal_range, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_saccade_vertical_range_distribution(real_number_eye_gaze_coordinates, saccade_index):
    saccade_vertical_range = []
    for saccade in saccade_index:
        saccade_vertical_range.append(
            max([coordinate[1] for coordinate in real_number_eye_gaze_coordinates[saccade[0]:saccade[1]]]) -
            min([coordinate[1] for coordinate in real_number_eye_gaze_coordinates[saccade[0]:saccade[1]]]))
    num_bins = 100
    counts, bin_edges = np.histogram(saccade_vertical_range, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_fixation_duration_distribution(fixation_index):
    fixation_durations = []
    for fixation in fixation_index:
        fixation_durations.append(fixation[1] - fixation[0])
    num_bins = 100
    counts, bin_edges = np.histogram(fixation_durations, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_fixation_start_time_distribution(fixation_index):
    fixation_start_time = []
    for fixation in fixation_index:
        fixation_start_time.append(fixation[0])
    num_bins = 100
    counts, bin_edges = np.histogram(fixation_start_time, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_fixation_horizontal_centroid_distribution(real_number_eye_gaze_coordinates, fixation_index):
    fixation_horizontal_centroid = []
    for fixation in fixation_index:
        fixation_horizontal_centroid.append(
            np.mean([coordinate[0] for coordinate in real_number_eye_gaze_coordinates[fixation[0]:fixation[1]]])
        )
    num_bins = 100
    counts, bin_edges = np.histogram(fixation_horizontal_centroid, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_fixation_vertical_centroid_distribution(real_number_eye_gaze_coordinates, fixation_index):
    fixation_vertical_centroid = []
    for fixation in fixation_index:
        fixation_vertical_centroid.append(
            np.mean([coordinate[1] for coordinate in real_number_eye_gaze_coordinates[fixation[0]:fixation[1]]])
        )
    num_bins = 100
    counts, bin_edges = np.histogram(fixation_vertical_centroid, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_blink_duration_distribution(blink_index):
    blink_durations = []
    for blink in blink_index:
        blink_durations.append(blink[1] - blink[0])
    num_bins = 100
    counts, bin_edges = np.histogram(blink_durations, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result


def get_blink_start_time_distribution(blink_index):
    blink_start_time = []
    for blink in blink_index:
        blink_start_time.append(blink[0])
    num_bins = 100
    counts, bin_edges = np.histogram(blink_start_time, bins=num_bins)
    result = [[0 for _ in range(100)] for _ in range(2)]
    for i in range(len(counts)):
        result[0][i] = bin_edges[i + 1]
        result[1][i] = counts[i]
    return result
