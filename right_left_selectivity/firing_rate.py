import numpy as np
from get_data import get_behavior, get_channels, get_spikes
def firingRates_onDepths(stim_events, spike_times, spike_depths, t_bin=0.1, d_bin=20, pre_stim=0.4, post_stim=1, depth_lim=[0, 3840]):
    """
    Calculate the z-scored firing rates on different depths for given stimulus events.
    Parameters:
    -----------
    stim_events : dict
        Dictionary where keys are stimulus types and values are arrays of stimulus onset times.
    spike_times : array-like
        Array of spike times.
    spike_depths : array-like
        Array of spike depths corresponding to the spike times.
    t_bin : float, optional
        Time bin size in seconds for binning spike times. Default is 0.1.
    d_bin : float, optional
        Depth bin size in micrometers for binning spike depths. Default is 20.
    pre_stim : float, optional
        Time in seconds before stimulus onset to include in the analysis and use as baseline period. Default is 0.4.
    post_stim : float, optional
        Time in seconds after stimulus onset to include in the analysis. Default is 1.
    depth_lim : list, optional
        List specifying the depth limits [min_depth, max_depth] in micrometers. Default is [0, 3840].
    Returns:
    --------
    z_scores : dict
        Dictionary where keys are stimulus types and values are 3D numpy arrays of z-scored firing rates 
        with shape (trials, depths, time).
    time : numpy.ndarray
        Array of time bins.
    depth : numpy.ndarray
        Array of depth bins.
    """


    from iblutil.numerical import bincount2D
    # Generate time and depth arrays
    time = np.arange(-pre_stim, post_stim +  t_bin / 2, t_bin)
    depth = np.arange(depth_lim[0], depth_lim[1] + d_bin, d_bin)
    
    z_scores = {stim_type: [] for stim_type in stim_events.keys()}

    for stim_type, stim_times in stim_events.items():
        stim_times = stim_times[~np.isnan(stim_times)]
        for stim_on_time in stim_times:
            interval = [stim_on_time - pre_stim, stim_on_time + post_stim]
            idx = np.where((spike_times > interval[0]) & (spike_times < interval[1]))[0]
            spike_times_i = spike_times[idx]
            spike_depths_i = spike_depths[idx]

            # Create binned array
            binned_array, tim, depths = bincount2D(spike_times_i, spike_depths_i, t_bin, d_bin, ylim= depth_lim, xlim=interval)

            # Compute baseline mean and std
            baseline_mean = np.mean(binned_array[:, :int(pre_stim/t_bin)], axis=1)
            baseline_std = np.std(binned_array[:, :int(pre_stim/t_bin)], axis=1)
            baseline_std[baseline_std == 0] = 1  # Avoid division by zero

            # Compute Z-score of firing rates
            z_score_firing_rate = (binned_array - baseline_mean[:, np.newaxis]) / baseline_std[:, np.newaxis]

            # Append to the respective list
            z_scores[stim_type].append(z_score_firing_rate)

    # Convert lists to numpy arrays for each stimulus type for final shape (trials, depths, time)
    z_scores = {stim_type: np.array(z_score_list) for stim_type, z_score_list in z_scores.items()}
    
    return z_scores, time, depth

def firing_Rates_onClusters(stim_events, spike_times, spike_clusters, t_bin=0.1, pre_stim=0.4, post_stim=1):
    """
    Calculate the z-scored firing rates of clusters around stimulus events.
    Parameters:
    -----------
    stim_events : dict
        Dictionary where keys are stimulus types and values are arrays of stimulus times.
    spike_times : array-like
        Array of spike times.
    spike_clusters : array-like
        Array of cluster IDs corresponding to each spike time.
    t_bin : float, optional
        Time bin size in seconds for binning spike times. Default is 0.1 seconds.
    pre_stim : float, optional
        Time in seconds before stimulus onset to include in the analysis and use as baseline. Default is 0.4 seconds.
    post_stim : float, optional
        Time in seconds after stimulus onset to include in the analysis. Default is 1 second.
    Returns:
    --------
    z_scores : dict
        Dictionary where keys are stimulus types and values are numpy arrays of z-scored firing rates 
        with shape (trials, clusters, time).
    time : numpy.ndarray
        Array of time points corresponding to the binned spike times.
    clusters : numpy.ndarray
        Array of unique cluster IDs.
    """
 
    from iblutil.numerical import bincount2D
    # Generate time array
    time = np.arange(-pre_stim, post_stim +  t_bin / 2, t_bin)
    clusters = np.unique(spike_clusters)
    
    z_scores = {stim_type: [] for stim_type in stim_events.keys()}

    for stim_type, stim_times in stim_events.items():
        stim_times = stim_times[~np.isnan(stim_times)] # Remove NaNs
        for stim_on_time in stim_times:
            interval = [stim_on_time - pre_stim, stim_on_time + post_stim]
            idx = np.where((spike_times > interval[0]) & (spike_times < interval[1]))[0]
            spike_times_i = spike_times[idx]
            spike_clusters_i = spike_clusters[idx]

            # Create binned array
            binned_array, tim, clusters = bincount2D(spike_times_i, spike_clusters_i, xbin= t_bin, ybin= 0, xlim=interval, ylim=clusters)

            # Compute baseline mean and std
            baseline_mean = np.mean(binned_array[:, :int(pre_stim/t_bin)], axis=1)
            baseline_std = np.std(binned_array[:, :int(pre_stim/t_bin)], axis=1)
            baseline_std[baseline_std == 0] = 1 # Avoid division by zero

            # Compute Z-score of firing rates
            z_score_firing_rate = (binned_array - baseline_mean[:, np.newaxis]) / baseline_std[:, np.newaxis]

            # Append to the respective list
            z_scores[stim_type].append(z_score_firing_rate)

    # Convert lists to numpy arrays for each stimulus type for final shape (trials, clusters, time)
    z_scores = {stim_type: np.array(z_score_list) for stim_type, z_score_list in z_scores.items()}

    return z_scores, time, clusters

def right_left_firingRates_onDepths(eid, pid, t_bin=0.1, d_bin=20, pre_stim=0.4, post_stim=1, depth_lim=[0, 3840], mode='download'):
    """
    Compute the z-scored firing rates on different depths for right and left stimulus onsets.
    Parameters:
    eid (str): Experiment ID.
    pid (str): Probe ID.
    t_bin (float, optional): Time bin size in seconds. Default is 0.1.
    d_bin (int, optional): Depth bin size in micrometers. Default is 20.
    pre_stim (float, optional): baseline - Time before stimulus onset to include in analysis  (in seconds). Default is 0.4.
    post_stim (float, optional): Time after stimulus onset to include in analysis (in seconds). Default is 1.
    depth_lim (list, optional): Depth limits to include in analysis (in micrometers). Default is [0, 3840].
    mode (str, optional): Mode to load data, either 'download' or 'load'. Default is 'download'.
    Returns:
    tuple: A tuple containing:
        - z_score_right (np.ndarray): Z-scored firing rates for right stimulus onset.
        - z_score_left (np.ndarray): Z-scored firing rates for left stimulus onset.
        - times (np.ndarray): Time points corresponding to the firing rates.
        - depths (np.ndarray): Depths corresponding to the firing rates.
        - ids (np.ndarray): Atlas IDs of the channels.
        - acronyms (np.ndarray): Acronyms of the brain regions corresponding to the channels.
    """
 

    ######################
    # Load data
    ########################
    # Load behavior
    behavior = get_behavior(eid, mode=mode)
    right_onset = behavior[behavior['contrastRight'] == 1]['stimOn_times']
    left_onset = behavior[behavior['contrastLeft'] == 1]['stimOn_times']
    stim_events = {'right': right_onset, 'left': left_onset}

    # load channel data 
    channels = get_channels(pid, eid,  mode=mode) # download the data from the IBL database or 'load' data from the local directory
    ids = []
    acronyms = []
    true_depths = []
    ch_indexs = []
    for depth in depths:
        id = channels[channels['axial_um'] == depth]['atlas_id']
        acronym = channels[channels['axial_um'] == depth]['acronym']
        ch_index = channels[channels['axial_um'] == depth].index
        if len(id) > 0:
            id = id.values[0]
            acro = acronym.values[0]
            ch_index = ch_index[0]
            ch_indexs.append(ch_index)
            acronyms.append(acro)
            ids.append(id)
            true_depths.append(depth)
    ids = np.array(ids)
    acronyms = np.array(acronyms)
    ch_indexs = np.array(ch_indexs)
    # load spikes
    spikes_datasets = get_spikes(pid, mode=mode)
    spikes = spikes_datasets['spikes']
    spike_times = spikes['times']
    spike_depths = spikes['depths']
    kp_idx = np.where(~np.isnan(spike_depths))[0] # Remove any nan depths
    spike_times = spike_times[kp_idx]
    spike_depths = spike_depths[kp_idx]

    ######################
    # compute firing rate
    ########################
    z_score_firing_rate, times, depths = firingRates_onDepths(stim_events, spike_times, spike_depths, t_bin = t_bin, d_bin=d_bin, pre_stim=pre_stim, post_stim=post_stim, depth_lim=depth_lim)
    z_score_right = z_score_firing_rate['right']
    z_score_left = z_score_firing_rate['left']

    return z_score_right, z_score_left,  times, depths, ids, acronyms, ch_indexs