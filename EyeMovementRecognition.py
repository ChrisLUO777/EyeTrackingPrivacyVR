import math
import numpy as np
from scipy import signal
import pandas as pd


def get_step_distance(coordinate0, coordinate1):
    step_distance = math.sqrt((coordinate1[1] - coordinate0[1])**2 + (coordinate1[0] - coordinate0[0])**2)
    return step_distance


def get_saccade_index(eye_movement_speed, peak_saccade_velocity_threshold):
    # return the start and end indexes of saccade periods.
    saccades = []
    saccade_start = 0
    above_threshold = False
    for saccade_end in range(len(eye_movement_speed)):
        if eye_movement_speed[saccade_end] > peak_saccade_velocity_threshold:
            above_threshold = True
        elif eye_movement_speed[saccade_end] <= peak_saccade_velocity_threshold:
            if above_threshold:
                saccades.append([saccade_start, saccade_end])
            saccade_start = saccade_end
            above_threshold = False
        if saccade_end == len(eye_movement_speed) - 1 and above_threshold:
            saccades.append([saccade_start, saccade_end])
    return saccades


def get_fixation_index(eye_movement_speed, peak_saccade_velocity_threshold):
    # return the start and end indexes of fixation periods.
    fixations = []
    fixation_start = 0
    below_threshold = False
    for fixation_end in range(len(eye_movement_speed)):
        if eye_movement_speed[fixation_end] <= peak_saccade_velocity_threshold:
            below_threshold = True
        elif eye_movement_speed[fixation_end] > peak_saccade_velocity_threshold:
            if below_threshold:
                fixations.append([fixation_start, fixation_end])
            fixation_start = fixation_end
            below_threshold = False
        if fixation_end == len(eye_movement_speed) - 1 and below_threshold:
            fixations.append([fixation_start, fixation_end])
    return fixations


def get_optimal_peak_saccade_velocity_threshold(eye_movement_speed):
    previous_peak_saccade_velocity_threshold = 0
    optimal_peak_saccade_velocity_threshold = 100
    while math.fabs(optimal_peak_saccade_velocity_threshold - previous_peak_saccade_velocity_threshold) > 1:
        previous_peak_saccade_velocity_threshold = optimal_peak_saccade_velocity_threshold
        fixation_periods = get_fixation_index(eye_movement_speed, previous_peak_saccade_velocity_threshold)
        fixation_velocities = []
        for fixation_index in range(len(fixation_periods)):
            fixation_velocities.extend(
                eye_movement_speed[fixation_periods[fixation_index][0]:fixation_periods[fixation_index][1]])
        fixation_velocities.sort()
        median = np.median(fixation_velocities)
        Q1 = np.percentile(fixation_velocities, 25, interpolation='midpoint')   # First quartile (Q1)
        Q3 = np.percentile(fixation_velocities, 75, interpolation='midpoint')   # Third quartile (Q3)
        IQR = Q3 - Q1   # Interquaritle range (IQR)
        optimal_peak_saccade_velocity_threshold = median + IQR
    return optimal_peak_saccade_velocity_threshold


def get_smooth_eye_movement_speed(eye_coordinates):
    eye_speed = []
    for step_index in range(len(eye_coordinates) - 1):
        # the sampling rate is 250 Hz
        cur_speed = get_step_distance(eye_coordinates[step_index], eye_coordinates[step_index + 1]) * 250
        if math.isnan(cur_speed):  # if the user is blinking, the derived speed will be "nan".
            eye_speed.append(0)  # replace "nan" with "0"
        else:
            eye_speed.append(cur_speed)
    smooth_eye_speed = []  # remove the spikes from the data using filters.
    win_len = 500
    for speed_index in range(len(eye_speed) - win_len):
        tmp_sum = sum(eye_speed[speed_index: speed_index + win_len])
        smooth_eye_speed.append(tmp_sum / win_len)
    return smooth_eye_speed
