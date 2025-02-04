# Get behavioral data
import numpy as np
import pandas as pd
import re
from .firing_rate_onCluster import firingRate_OnClusters
from one.api import ONE
one = ONE(base_url='https://openalyx.internationalbrainlab.org')

import sys
from pathlib import Path
import os

from extraction_data import get_behavior, get_spikes, get_channels
def pre_processed_passive_data(eid, pid, **kwargs):
    
    min_contrast = kwargs.get('min_contrast', 0.25)
    t_bin = kwargs.get('t_bin', 0.02)
    pre_stim = kwargs.get('pre_stim', 0.5)
    post_stim = kwargs.get('post_stim', 1.0)
    min_time = kwargs.get('min_time', 0) * 1000 # Convert to milliseconds
    filter_regions = kwargs.get('filter_regions', None)
    only_good_clusters = kwargs.get('only_good_clusters', True)
    contrast_filter = kwargs.get('contrast_stim_filter', [0, 0.25, 1])
    ######################################################
    # Extract trial information                                
    ######################################################
    # Load passive Gabor data
    passiveGabor = one.load_object(eid, 'passiveGabor')
    passiveGabor = pd.DataFrame(passiveGabor)

    # filter based on contrast  
    valid_trials= np.where(np.isin(passiveGabor['contrast'], contrast_filter))[0]

    behavior = passiveGabor.loc[valid_trials].reset_index(drop=True)

    # trial metadata
    trial_indx = valid_trials 
    trial_onsets = behavior['start'].values
    contrasts = behavior['contrast'].values
    phases = behavior['phase'].values
    positions = behavior['position'].values
    # Assign labels: 1 for right (+35), 0 for left (-35)
    index_right = np.where(positions == 35)[0]
    index_left = np.where(positions == -35)[0]
    index_no_stim = np.where(contrasts == 0)[0]
    labels = np.full(len(trial_indx), np.nan)
    labels[index_right] = 1
    labels[index_left] = -1
    labels[index_no_stim] = 0
    

    ######################################################
    # Load spikes data                        
    ######################################################

    spike_activity = get_spikes(pid, modee='download')
    channels = get_channels(eid, pid, modee='download')
    clusters = spike_activity['clusters']
    spikes = spike_activity['spikes']
    channels_clusters = clusters['channels'] # each cluster is assigned to which channel 
    spike_times, spike_clusters = spikes['times'], spikes['clusters'] # Get the spike times and clusters

    # remove nan clusters
    kp_idx = np.where(~np.isnan(spike_clusters))[0]
    spike_times, spike_clusters = spike_times[kp_idx], spike_clusters[kp_idx]

    ## Filter out Bad clusters 
    if only_good_clusters:
        metrics = clusters['metrics'].reset_index(drop=True)
        good_clusters = np.where(metrics['label'] > 0.6)[0]

    else:
        good_clusters= np.unique(spike_clusters)

    # filter cluster based on regions
    if filter_regions:
        # Filter channels based on brain regions
        # clusters = np.unique(channels_clusters)
        index_channel = [i for i, acronym in enumerate(channels['acronym']) for region in filter_regions if re.match(rf'^{region}[12456]', acronym) and i in channels_clusters]
        region_clusters = np.where(np.isin(channels_clusters, index_channel))[0]

    else:
        index_channel = np.unique(channels_clusters)
        region_clusters = np.unique(spike_clusters)

    selected_clusters = np.intersect1d(good_clusters, region_clusters)
    keep_indices = np.where(np.isin(spike_clusters, selected_clusters))[0]
    spike_clusters = spike_clusters[keep_indices]
    spike_times = spike_times[keep_indices]
    channels_clusters = channels_clusters[selected_clusters]
    index_channel = np.intersect1d(channels_clusters, index_channel)
    
    # z_score_firing_rate.shape = (n_trials, n_clusters, n_time_bins) | times.shape = (n_time_bins,) in milliseconds | clusters.shape = (n_clusters,)
    z_score_firing_rate, times, clusters = firingRate_OnClusters(trial_onsets, spike_times, spike_clusters, t_bin=t_bin, pre_stim=pre_stim, post_stim=post_stim)

    # Filter time indices to include only those after time 0
    time_indices = np.where(times >= min_time)[0]
    times = times[time_indices]
    z_score_firing_rate = z_score_firing_rate[:, :, time_indices]

    # save Firing rates for each channel                     
    FR_channel = {}
    nan_channels = []
    for i, ch in enumerate(index_channel):
        indx_cluster = np.where(channels_clusters == ch)[0]
        if len(indx_cluster) > 0:
            FR_channel[ch] = z_score_firing_rate[:, indx_cluster, :]


    #############################
    # Extract channel metadata
    ##############################

    ids, acronyms, depths, ch_indexs, coordinates = [], [], [], [], []
    for ch in index_channel:
        channel_info = channels.loc[ch]
        ids.append(channel_info['atlas_id'])
        acronyms.append(channel_info['acronym'])
        ch_indexs.append(ch)
        coordinates.append(channel_info[['x', 'y', 'z']])
        depths.append(channel_info['axial_um'])
    ids, acronyms, ch_indexs, coordinates = np.array(ids), np.array(acronyms), np.array(ch_indexs), np.array(coordinates)


    # save channel metadata and trial metadata into dataframes
    channel_info = pd.DataFrame({'depth': depths, 'ids': ids, 'acronyms': acronyms, 'ch_indexs': ch_indexs, 'x_coordinates': coordinates[:, 0], 'y_coordinates': coordinates[:, 1], 'z_coordinates': coordinates[:, 2]})
    trial_info = pd.DataFrame({'trial_index': trial_indx, 'labels': labels, 'assigned_side': np.full(len(trial_indx), np.nan), 'contrasts': contrasts, 'distance_to_change': np.full(len(trial_indx), np.nan), 'prob_left': np.full(len(trial_indx), np.nan), 'probe_id': np.repeat(pid, trial_indx.size), 'experiment_id': np.repeat(eid, trial_indx.size), 'phase': phases})

    # final preprocessed data
    pre_processed_data = {'firing_rates': FR_channel, 'trial_info': trial_info, 'channel_info': channel_info, 'time_bins': times}
   
    return pre_processed_data
