import numpy as np
from functions import bincount2D_cluster
import pandas as pd

def firingRate_OnClusters(stim_times,spike_times, spike_clusters, t_bin=0.1, pre_stim=0.4, post_stim=1, z_score=True):

    '''
    make sure the spike data do not contain nan values
    if you want to include onlu the good clusters make sure in the previous step you removed the bad clusters
    
    
    '''
    # convert time to milliseconds to avoid floating point errors
    stim_times, spike_times, pre_stim, post_stim, t_bin = [x * 1000 for x in [stim_times, spike_times, pre_stim, post_stim, t_bin]]
    all_clusters = np.unique(spike_clusters)  # Global list of all clusters
    z_scores = []
    for stim_on_time in stim_times:
        interval = [stim_on_time - pre_stim, stim_on_time + post_stim]
        idx = np.where((spike_times > interval[0]) & (spike_times < interval[1]))[0]
        spike_times_i = spike_times[idx]
        spike_clusters_i = spike_clusters[idx]

        # Bin spike data using the  bincoun2D_cluster function
        binned_array, tim, cluster = bincount2D_cluster(
            x=spike_times_i,
            y=spike_clusters_i,
            xbin=t_bin,
            ybin=0, 
            xlim=interval,
            yscale=all_clusters  
        )
        if z_score == False:
            z_scores.append(binned_array)
            continue
        # Calculate baseline mean and std for Z-score calculation
        baseline_mean = np.mean(binned_array[:, :int(pre_stim / t_bin)], axis=1)
        baseline_std = np.std(binned_array[:, :int(pre_stim / t_bin)], axis=1)
        baseline_std[baseline_std == 0] = 1

        # Z-score firing rates and append
        z_score_firing_rate = (binned_array - baseline_mean[:, np.newaxis]) / baseline_std[:, np.newaxis]
        z_scores.append(z_score_firing_rate)
    z_scores = np.array(z_scores)
    times = tim - stim_on_time

    return z_scores, times, all_clusters

