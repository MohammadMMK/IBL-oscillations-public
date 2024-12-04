import numpy as np
import sys
from pathlib import Path
import pandas as pd
from iblutil.numerical import bincount2D



def firingRates_onDepths(stim_events, spike_times, spike_depths, t_bin=0.1, d_bin=20, pre_stim=0.4, post_stim=1):
    """
    Calculate z-scored firing rates binned by time and depth.

    Parameters:
    - stim_events: dict, stimulus onset times for different conditions
    - spike_times: array, times of all spikes
    - spike_depths: array, depths of all spikes
    - t_bin: float, time bin size in seconds
    - d_bin: float, depth bin size in micrometers
    - pre_stim: float, time before stimulus onset to include in the analysis
    - post_stim: float, time after stimulus onset to include in the analysis

    Returns:
    - z_scores: dict, z-scored firing rates for each condition (trials, depths, time)
    - time_centers: array, time bin centers (relative to stimulus onset)
    - depth_centers: array, depth bin centers
    """
    spike_times = spike_times * 1000  # Convert to milliseconds
    depth_lim = [10, 3840]  # Fixed depth limits for this dataset
    z_scores = {stim_type: [] for stim_type in stim_events.keys()}
    pre_stim = pre_stim * 1000  # Convert to milliseconds
    post_stim = post_stim * 1000  # Convert to milliseconds
    t_bin = t_bin * 1000  # Convert to milliseconds
    for stim_type, stim_times in stim_events.items():
        stim_times = stim_times[~np.isnan(stim_times)]
        stim_times = stim_times * 1000  # Convert to milliseconds
        for stim_on_time in stim_times:
            interval = [stim_on_time - pre_stim, stim_on_time + post_stim]
            idx = np.where((spike_times > interval[0]) & (spike_times < interval[1]))[0]
            spike_times_i = spike_times[idx]
            spike_depths_i = spike_depths[idx]

            # Bin spike data by time and depth
            binned_array, tim, depths = bincount2D(spike_times_i, spike_depths_i, xbin=t_bin, ybin=d_bin, xlim=interval, ylim=depth_lim)

            # Calculate baseline mean and std for Z-score calculation
            baseline_mean = np.mean(binned_array[:, :int(pre_stim / t_bin)], axis=1)
            baseline_std = np.std(binned_array[:, :int(pre_stim / t_bin)], axis=1)
            baseline_std[baseline_std == 0] = 1  # Avoid division by zero

            # Z-score firing rates and append
            z_score_firing_rate = (binned_array - baseline_mean[:, np.newaxis]) / baseline_std[:, np.newaxis]
            z_scores[stim_type].append(z_score_firing_rate)

    # Convert lists to numpy arrays
    z_scores = {stim_type: np.array(z_score_list) for stim_type, z_score_list in z_scores.items()}

    time_centers = tim - stim_on_time
    depth_centers = depths + d_bin / 2

    return z_scores, time_centers, depth_centers