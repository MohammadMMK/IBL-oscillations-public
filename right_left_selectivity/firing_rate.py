import numpy as np
from get_data import get_behavior, get_channels, get_spikes
from iblutil.numerical import bincount2D

def firingRates_onDepths(stim_events, spike_times, spike_depths, t_bin=0.1, d_bin=20, pre_stim=0.4, post_stim=1, depth_lim=[10, 3840]):
    """
    Calculate Z-scored firing rates on depth bins for specified stimulus events.

    Parameters:
    - stim_events (dict): Dictionary with keys as stimulus types and values as arrays of stimulus onset times.
    - spike_times (np.ndarray): Array of spike times.
    - spike_depths (np.ndarray): Array of spike depths.
    - t_bin (float): Time bin size in seconds.
    - d_bin (int): Depth bin size in micrometers.
    - pre_stim (float): Time in seconds before stimulus onset.
    - post_stim (float): Time in seconds after stimulus onset.
    - depth_lim (list): Depth range for analysis in micrometers.

    Returns:
    - z_scores (dict): Dictionary of Z-scored firing rates for each stimulus type, shape (trials, depths, time).
    - time_centers (np.ndarray): Time bins centered around stimulus onset.
    - depth_centers (np.ndarray): Depth bins for analysis.
    """
    z_scores = {stim_type: [] for stim_type in stim_events.keys()}

    for stim_type, stim_times in stim_events.items():
        stim_times = stim_times[~np.isnan(stim_times)]
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


def firing_Rates_onClusters(stim_events, spike_times, spike_clusters, t_bin=0.1, pre_stim=0.4, post_stim=1):
    """
    Calculate Z-scored firing rates on clusters for specified stimulus events.

    Parameters:
    - stim_events (dict): Dictionary with keys as stimulus types and values as arrays of stimulus onset times.
    - spike_times (np.ndarray): Array of spike times.
    - spike_clusters (np.ndarray): Array of spike cluster IDs.
    - t_bin (float): Time bin size in seconds.
    - pre_stim (float): Time in seconds before stimulus onset.
    - post_stim (float): Time in seconds after stimulus onset.

    Returns:
    - z_scores (dict): Dictionary of Z-scored firing rates for each stimulus type, shape (trials, clusters, time).
    - times (np.ndarray): Time bins centered around stimulus onset.
    - clusters (np.ndarray): Unique cluster IDs.
    """
    clusters = np.unique(spike_clusters)
    z_scores = {stim_type: [] for stim_type in stim_events.keys()}

    for stim_type, stim_times in stim_events.items():
        stim_times = stim_times[~np.isnan(stim_times)]
        for stim_on_time in stim_times:
            interval = [stim_on_time - pre_stim, stim_on_time + post_stim]
            idx = np.where((spike_times > interval[0]) & (spike_times < interval[1]))[0]
            spike_times_i = spike_times[idx]
            spike_clusters_i = spike_clusters[idx]

            # Bin spike data by time and clusters
            binned_array, tim, clusters = bincount2D(spike_times_i, spike_clusters_i, xbin=t_bin, ybin=0, xlim=interval, ylim=clusters)

            # Calculate baseline mean and std for Z-score calculation
            baseline_mean = np.mean(binned_array[:, :int(pre_stim / t_bin)], axis=1)
            baseline_std = np.std(binned_array[:, :int(pre_stim / t_bin)], axis=1)
            baseline_std[baseline_std == 0] = 1

            # Z-score firing rates and append
            z_score_firing_rate = (binned_array - baseline_mean[:, np.newaxis]) / baseline_std[:, np.newaxis]
            z_scores[stim_type].append(z_score_firing_rate)

    # Convert lists to numpy arrays
    z_scores = {stim_type: np.array(z_score_list) for stim_type, z_score_list in z_scores.items()}
    times = tim - stim_on_time
    return z_scores, times, clusters


def right_left_firingRates_onDepths(eid, pid, t_bin=0.1, d_bin=20, pre_stim=0.4, post_stim=1, depth_lim=[10, 3840], modee='download'):
    """
    Compute firing rates for right and left stimulus events on depth bins.

    Parameters:
    - eid (str): Experiment ID.
    - pid (str): Probe insertion ID.
    - t_bin (float): Time bin size in seconds.
    - d_bin (int): Depth bin size in micrometers.
    - pre_stim (float): Time in seconds before stimulus onset.
    - post_stim (float): Time in seconds after stimulus onset.
    - depth_lim (list): Depth range for analysis in micrometers.
    - modee (str): Mode of data loading, either 'download' or 'load' from local directory.

    Returns:
    - z_score_right (np.ndarray): Z-scored firing rates for right stimulus trials.
    - z_score_left (np.ndarray): Z-scored firing rates for left stimulus trials.
    - times (np.ndarray): Time bins centered around stimulus onset.
    - depths (np.ndarray): Depth bins for analysis.
    - ids, acronyms, ch_indexs, coordinates (np.ndarrays): Channel metadata.
    - trial_indx, distance_to_change, probs_left (np.ndarrays): Behavioral metadata.
    """
    # Load behavior data
    behavior = get_behavior(eid, modee=modee)
    right_onset = behavior[behavior['contrastRight'] == 1]['stimOn_times']
    left_onset = behavior[behavior['contrastLeft'] == 1]['stimOn_times']
    stim_events = {'right': right_onset, 'left': left_onset}
    indx_right, indx_left = list(right_onset.index), list(left_onset.index)
    trial_indx = np.concatenate((indx_right, indx_left))

    # Calculate distance to last block change
    change_indices = behavior['probabilityLeft'].ne(behavior['probabilityLeft'].shift()).to_numpy().nonzero()[0]
    distance_to_change = np.array([0 if i in change_indices else i - change_indices[change_indices < i][-1] for i in range(len(behavior))])
    distance_to_change = distance_to_change[trial_indx]
    probs_left = behavior['probabilityLeft'][trial_indx].reset_index(drop=True)

    # Load channels and spikes data
    channels = get_channels(eid, pid, modee=modee)
    spikes = get_spikes(pid, modee=modee)['spikes']
    spike_times, spike_depths = spikes['times'], spikes['depths']
    kp_idx = np.where(~np.isnan(spike_depths))[0]
    spike_times, spike_depths = spike_times[kp_idx], spike_depths[kp_idx]

    # Compute Z-scored firing rates on depths
    z_score_firing_rate, times, depths = firingRates_onDepths(stim_events, spike_times, spike_depths, t_bin, d_bin, pre_stim, post_stim, depth_lim)
    z_score_right, z_score_left = z_score_firing_rate['right'], z_score_firing_rate['left']

    # Extract channel metadata
    ids, acronyms, true_depths, ch_indexs, coordinates = [], [], [], [], []
    for depth in depths:
        channel_info = channels[channels['axial_um'] == depth]
        if not channel_info.empty:
            ids.append(channel_info['atlas_id'].values[0])
            acronyms.append(channel_info['acronym'].values[0])
            ch_indexs.append(channel_info.index[0])
            coordinates.append(channel_info[['x', 'y', 'z']].values[0])
            true_depths.append(depth)

    return (z_score_right, z_score_left, times, depths, np.array(ids), np.array(acronyms), 
            np.array(ch_indexs), np.array(coordinates), trial_indx, distance_to_change, probs_left)
