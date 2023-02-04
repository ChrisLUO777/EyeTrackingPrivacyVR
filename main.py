# This is the main program to segment and extract features from the eye movement data.
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pandas as pd
import collections

from EyeMovementRecognition import get_smooth_eye_movement_speed
from EyeMovementRecognition import get_optimal_peak_saccade_velocity_threshold
from EyeMovementRecognition import get_saccade_index
from SaccadeFeatureExtraction import get_saccade_duration_distribution

if __name__ == '__main__':
    df = pd.read_csv('./gazebasevr/data/S_1002_S1_3_VID.csv')
    left_eye_coordinates = [coordinate for coordinate in list(zip(df.get("lx"), df.get("ly")))]
    smooth_left_eye_speed = get_smooth_eye_movement_speed(left_eye_coordinates)
    updated_peak_saccade_velocity_threshold = get_optimal_peak_saccade_velocity_threshold(smooth_left_eye_speed)
    print(updated_peak_saccade_velocity_threshold)
    saccade_index = get_saccade_index(smooth_left_eye_speed, updated_peak_saccade_velocity_threshold)
    saccade_durations = get_saccade_duration_distribution(saccade_index)

    # x_axis = [i*4 for i in range(len(smooth_left_eye_speed))]
    # plt.plot(x_axis, smooth_left_eye_speed)
    # plt.plot(x_axis, [updated_peak_saccade_velocity_threshold] * len(x_axis))
    # plt.title('Left Eye Movement')
    # plt.xlabel('msec')
    # plt.ylabel('Eye Movement Speed (deg/sec)')
    # plt.show()
