import numpy as np
import pandas as pd 
from get_data import get_behavior, get_channels, get_spikes
from iblutil.numerical import bincount2D
# Import necessary libraries
import numpy as np
import sys
from pathlib import Path
import pandas as pd

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

# def firing_Rates_onClusters(stim_events, spike_times, spike_clusters, t_bin=0.1, pre_stim=0.4, post_stim=1):
#     """
#     Calculate z-scored firing rates binned by time and cluster.

#     Parameters:
#     - stim_events: dict, stimulus onset times for different conditions
#     - spike_times: array, times of all spikes
#     - spike_clusters: array, cluster assignments of all spikes
#     - t_bin: float, time bin size in seconds
#     - pre_stim: float, time before stimulus onset to include in the analysis
#     - post_stim: float, time after stimulus onset to include in the analysis

#     Returns:
#     - z_scores: dict, z-scored firing rates for each condition (trials, clusters, time)
#     - times: array, time bin centers (relative to stimulus onset)
#     - clusters: array, unique cluster identifiers
#     """
#     clusters = np.unique(spike_clusters)
#     z_scores = {stim_type: [] for stim_type in stim_events.keys()}

#     for stim_type, stim_times in stim_events.items():
#         stim_times = stim_times[~np.isnan(stim_times)]
#         for stim_on_time in stim_times:
#             interval = [stim_on_time - pre_stim, stim_on_time + post_stim]
#             idx = np.where((spike_times > interval[0]) & (spike_times < interval[1]))[0]
#             spike_times_i = spike_times[idx]
#             spike_clusters_i = spike_clusters[idx]

#             # Bin spike data by time and clusters
#             binned_array, tim, clusters = bincount2D(spike_times_i, spike_clusters_i, xbin=t_bin, ybin=0, xlim=interval, ylim=clusters)

#             # Calculate baseline mean and std for Z-score calculation
#             baseline_mean = np.mean(binned_array[:, :int(pre_stim / t_bin)], axis=1)
#             baseline_std = np.std(binned_array[:, :int(pre_stim / t_bin)], axis=1)
#             baseline_std[baseline_std == 0] = 1

#             # Z-score firing rates and append
#             z_score_firing_rate = (binned_array - baseline_mean[:, np.newaxis]) / baseline_std[:, np.newaxis]
#             z_scores[stim_type].append(z_score_firing_rate)

#     # Convert lists to numpy arrays
#     z_scores = {stim_type: np.array(z_score_list) for stim_type, z_score_list in z_scores.items()}
#     times = tim - stim_on_time
#     return z_scores, times, clusters

def all_trials_firingRates_onDepths(eid, pid, t_bin=0.1, d_bin=20, pre_stim=0.4, post_stim=1, modee='download', min_contrast=0, probability_left='all', min_time = 0):

    # Get behavioral data
    behavior = get_behavior(eid, modee=modee)

    # Filter trials based on probability_left if specified
    if probability_left != 'all':
        behavior = behavior[behavior['probabilityLeft'] == probability_left]

    # Filter trials based on minimum contrast
    valid_trials = behavior[
        (behavior['contrastRight'] >= min_contrast) | (behavior['contrastLeft'] >= min_contrast)
    ]

    # Extract trial information
    trial_onsets = valid_trials['stimOn_times'].values
    contrast_right = valid_trials['contrastRight'].fillna(0).values
    contrast_left = valid_trials['contrastLeft'].fillna(0).values
    contrasts = np.maximum(contrast_right, contrast_left)  # Use maximum contrast for each trial

    # Assign labels: 1 for right (contrastRight > 0), 0 for left (contrastLeft > 0)
    labels = (contrast_right > 0).astype(int)

    # Calculate distance to the last block change
    change_indices = behavior['probabilityLeft'].ne(behavior['probabilityLeft'].shift()).to_numpy().nonzero()[0]
    distance_to_change = np.array([0 if i in change_indices else i - change_indices[change_indices < i][-1] for i in range(len(behavior))])
    distance_to_change = distance_to_change[valid_trials.index]

    # Get the bias blocks
    probs_left = valid_trials['probabilityLeft'].values

    # Load channels and spikes data
    channels = get_channels(eid, pid, modee=modee)
    spikes = get_spikes(pid, modee=modee)['spikes']
    spike_times, spike_depths = spikes['times'], spikes['depths']
    kp_idx = np.where(~np.isnan(spike_depths))[0]
    spike_times, spike_depths = spike_times[kp_idx], spike_depths[kp_idx]

    # Compute Z-scored firing rates on depths for all trials
    stim_events = {'all': trial_onsets}
    z_score_firing_rate, times, depths = firingRates_onDepths(stim_events, spike_times, spike_depths, t_bin, d_bin, pre_stim, post_stim)
    firing_rates = z_score_firing_rate['all']  # Shape: (trials, depths, time)

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

    trial_indx = valid_trials.index.values
    ids, acronyms, ch_indexs, coordinates = np.array(ids), np.array(acronyms), np.array(ch_indexs), np.array(coordinates)

    # Filter time indices to include only those after time 0
    min_time = min_time * 1000  # Convert to milliseconds
    time_indices = np.where(times >= min_time)[0]
    
    # Extract firing rates for all time bins after time 0
    firing_rates = firing_rates[:, :, time_indices]



    # stack depth information into one datafram 
    depth_info = pd.DataFrame({'depth': depths, 'ids': ids, 'acronyms': acronyms, 'ch_indexs': ch_indexs, 'x_coordinates': coordinates[:, 0], 'y_coordinates': coordinates[:, 1], 'z_coordinates': coordinates[:, 2]})
    # Repeat probe and experiment IDs to match trial data structure
    probe_ids = np.repeat(pid, trial_indx.size)
    experiment_ids = np.repeat(eid, trial_indx.size)
    trial_info = pd.DataFrame({'trial_index': trial_indx, 'labels': labels, 'contrasts': contrasts, 'distance_to_change': distance_to_change, 'prob_left': probs_left, 'probe_id': probe_ids, 'experiment_id': experiment_ids})

    # create a dictionary to store all the data
    data = {'firing_rates': firing_rates,  'trial_info': trial_info,  'depth_info': depth_info, 'time_bins': times[time_indices]}
    # Prepare output
    return data

def all_trials_firingRates_onDepths_passive(eid, pid, t_bin=0.1, d_bin=20, pre_stim=0.4, post_stim=1, modee='download', min_contrast=0, min_time = 0):

    from one.api import ONE
    import pandas as pd

    one = ONE(base_url='https://openalyx.internationalbrainlab.org')

    # Load passive Gabor data
    passiveGabor = one.load_object(eid, 'passiveGabor')
    passiveGabor = pd.DataFrame(passiveGabor)

    # Filter trials based on minimum contrast
    valid_trials = passiveGabor[passiveGabor['contrast'] >= min_contrast]
    
    # Extract trial information
    trial_onsets = valid_trials['start'].values
    contrasts = valid_trials['contrast'].values
    phases = valid_trials['phase'].values
    positions = valid_trials['position'].values

    # Assign labels: 1 for right (+35), 0 for left (-35)
    labels = (positions == +35).astype(int)

    # Load channels and spikes data
    channels = get_channels(eid, pid, modee=modee)
    spikes = get_spikes(pid, modee=modee)['spikes']
    spike_times, spike_depths = spikes['times'], spikes['depths']
    kp_idx = np.where(~np.isnan(spike_depths))[0]
    spike_times, spike_depths = spike_times[kp_idx], spike_depths[kp_idx]

    # Compute Z-scored firing rates on depths for all trials
    stim_events = {'all': trial_onsets}
    z_score_firing_rate, times, depths = firingRates_onDepths(stim_events, spike_times, spike_depths, t_bin, d_bin, pre_stim, post_stim)
    firing_rates = z_score_firing_rate['all']  # Shape: (trials, depths, time)
    # Filter time indices to include only those after time 0
    min_time = min_time * 1000  # Convert to milliseconds
    time_indices = np.where(times >= min_time)[0]
    # Extract firing rates for all time bins after time 0
    firing_rates = firing_rates[:, :, time_indices]

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
    ids, acronyms, ch_indexs, coordinates = np.array(ids), np.array(acronyms), np.array(ch_indexs), np.array(coordinates)

    # stack depth information into one datafram 
    depth_info = pd.DataFrame({'depth': depths, 'ids': ids, 'acronyms': acronyms, 'ch_indexs': ch_indexs, 'x_coordinates': coordinates[:, 0], 'y_coordinates': coordinates[:, 1], 'z_coordinates': coordinates[:, 2]})
    # Repeat probe and experiment IDs to match trial data structure

    trial_indx = valid_trials['Unnamed: 0']
    
    
    distance_to_change = np.full(len(trial_indx), np.nan)
    probs_left = np.full(len(trial_indx), np.nan)
    # Repeat probe and experiment IDs to match trial data structure
    probe_ids = np.repeat(pid, len(trial_indx))
    experiment_ids = np.repeat(eid, len(trial_indx))
    trial_info = pd.DataFrame({'trial_index': trial_indx, 'labels': labels, 'contrasts': contrasts, 'distance_to_change': distance_to_change, 'prob_left': probs_left, 'probe_id': probe_ids, 'experiment_id': experiment_ids})

    # create a dictionary to store all the data
    data = {'firing_rates': firing_rates,  'trial_info': trial_info,  'depth_info': depth_info, 'time_bins': times[time_indices]}

    return data




def firingRates_onDepths_passiveActive(pid_eid_pair, t_bin=0.025, pre_stim=0.8, post_stim=0.3,  modee = 'download', min_contrast=1, probability_left=0.5, min_time = 0, base_path = '/mnt/data/AdaptiveControl/IBLrawdata/classification/preprocess_firingRate', overwrite = False):
    import os
    import pickle


    # Initialize identifiers
    pid = pid_eid_pair[0]
    eid = pid_eid_pair[1]

    path_save = os.path.join(base_path, f'{pid}.pkl')
    if os.path.exists(path_save) and not overwrite:
        print(f'File {path_save} already exists. Skipping...')
        return 1

    print(f'Processing {pid} {eid} active')
    data_active = all_trials_firingRates_onDepths(eid, pid, t_bin=t_bin,  pre_stim=pre_stim, post_stim=post_stim, modee= modee, min_contrast=min_contrast, probability_left= probability_left, min_time = min_time)
    print(f'Processing {pid} {eid} passive')
    data_passive = all_trials_firingRates_onDepths_passive(eid, pid, t_bin=t_bin, pre_stim=pre_stim, post_stim=post_stim, modee= modee, min_contrast=min_contrast, min_time = min_time)
    print(f'Processing {pid} {eid} done')
    if data_active['firing_rates'].shape[2] != data_passive['firing_rates'].shape[2]:
        print(f'Active and passive data have different shapes for {pid} {eid}. Skipping...')
        print (data_active['time_bins'])
        print (data_passive['time_bins'])
        return 2
    # merge active and passive data
    FR_merged = np.concatenate((data_active['firing_rates'], data_passive['firing_rates']), axis=0) # concatenate along the trial   axis
    trial_info_merged = pd.concat([data_active['trial_info'], data_passive['trial_info']], axis=0)
    data_merged = {'firing_rates': FR_merged, 'trial_info': trial_info_merged, 'depth_info': data_active['depth_info'], 'time_bins': data_active['time_bins']}

    with open(path_save, 'wb') as f:
        pickle.dump(data_merged, f)
    return  0
