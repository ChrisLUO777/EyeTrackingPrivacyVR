import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob


def get_step_distance(coordinate0, coordinate1):
    step_distance = math.sqrt(min((360 - abs(coordinate1[0] - coordinate0[0]))**2, (coordinate1[0] - coordinate0[0])**2)
                              + (coordinate1[1] - coordinate0[1])**2)
    return step_distance


path = './scanpath/head trace'
extension = 'csv'
os.chdir(path)  # note this command change current directory into [path].
head_coordinate_files = glob.glob('*.{}'.format(extension))
head_movement_angle = []
head_movement_threshold = 5
segment_lengths_under_threshold = []

for head_coordinate_file in head_coordinate_files:
    df = pd.read_csv(head_coordinate_file)
    # note the column names start with a " "
    head_coordinates = [coordinate for coordinate in list(zip(df.get(" longitude") * 360, df.get(" latitude") * 180))]
    segment_length_under_threshold = 0
    for user_idx in range(57):
        segment_length_under_threshold = 0
        for step_index in range(user_idx * 100, user_idx * 100 + 99):
            # each frame is 0.2 sec long
            cur_speed = get_step_distance(head_coordinates[step_index], head_coordinates[step_index + 1])
            head_movement_angle.append(cur_speed)
            if cur_speed <= head_movement_threshold:
                segment_length_under_threshold += 1
            elif cur_speed > head_movement_threshold:   # note the program can only pick EITHER "if" or "elif" branch,
                if segment_length_under_threshold > 0:
                    segment_lengths_under_threshold.append(segment_length_under_threshold)
                segment_length_under_threshold = 0
            # so we have to deal with the "end of the line" case in another "if" branch.
            if step_index == user_idx * 100 + 98:
                if segment_length_under_threshold > 0:
                    segment_lengths_under_threshold.append(segment_length_under_threshold)
                segment_length_under_threshold = 0


# # plot the head movement speed vs time figure
# x_axis = [i * 200 for i in range(len(head_speed))]  # each frame is 200 ms long
# plt.plot(x_axis, head_speed)
# plt.title('Head Movement')
# plt.xlabel('msec')
# plt.ylabel('Head Movement Angle (deg)')
# plt.show()

# calculate the total length of segments under the threshold
print("ratio of time when head movement is less than 10 degrees: ",
      sum(segment_lengths_under_threshold) / (99 * 57 * 19))

# plot the distribution (PDF) of the head movement speed
num_bins = 100
counts, bin_edges = np.histogram(head_movement_angle, bins=num_bins)    # Use the histogram function to bin the data
plt.subplot(2, 1, 1)
plt.plot(bin_edges[1:], counts / len(head_movement_angle))    # And finally plot the pdf.
# Note bin_edges is 1 greater than num of bins.
plt.title('Head Movement Angle Distribution')
plt.ylabel('Probability')
cdf = np.cumsum(counts)
plt.subplot(2, 1, 2)
plt.plot(bin_edges[1:], cdf / len(head_movement_angle))    # plot the cdf
plt.xlabel('Head Movement Angle (deg)')
plt.ylabel('CDF')
plt.show()
