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
