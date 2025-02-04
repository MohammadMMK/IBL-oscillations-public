
import os
from pathlib import Path
import sys
from extraction_data import get_epoch_StimOn, get_behavior, get_channels, numpyToMNE
import numpy as np
import mne
from .split_trials import split_trials
from .csd import CSD_epoch




# def get_TFR_old(pid, ch_index, eid, n_jobs, freqs, n_cycles, time_bandwidth, get_LFP = 'load',  condition = 'Right_Left', tmin = -0.5, tmax = 1 , remove_first_trials_of_block = False, min_trial = 0, **kwargs):

#     # parameters
#     sfreq = kwargs.get('sfreq', 500)

    
#     sfreq =500
#     # load data
#     epochs_full = get_epoch_StimOn(pid, modee= get_LFP, ephys_path = paths["LFP"] )
#     epochs_full= epochs_full * 1e6
#     behavior = get_behavior(eid, modee='download')
#     channels = get_channels(eid,pid, modee = 'download')

#     # mne epochs
#     acronyms = channels['acronym'].tolist()
#     ch_names=[f'{ch}_{i}' for i, ch in enumerate(acronyms)]
#     epochs_mne = numpyToMNE(epochs_full, behavior, ch_names, sfreq = sfreq )

#     # filtering
#     epochs_mne = epochs_mne.crop(tmin=tmin, tmax=tmax)
#     epochs_mne_filter = epochs_mne.filter(l_freq= 0.5, h_freq= 55, n_jobs=n_jobs)

#     epochs_mne_filter = CSD_epoch(epochs_mne_filter, channels)

#     # select channels
#     ch_names = [f'{ch}_{i}' for i, ch in enumerate(acronyms) if i in ch_index]
#     epochs_mne_filter= epochs_mne_filter.pick_channels(ch_names= ch_names) 

#     # split trials
#     epochs_1, epochs_2 = split_trials(epochs_mne_filter, condition , min_trial = min_trial, remove_first_trials_of_block = remove_first_trials_of_block)

#     ######################
#     # TFR

#     averageTFR_1 = mne.time_frequency.tfr_multitaper(epochs_1, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, return_itc=False, average=True, n_jobs= n_jobs)
#     averageTFR_2 = mne.time_frequency.tfr_multitaper(epochs_2, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, return_itc=False, average=True, n_jobs= n_jobs)
    
#     return averageTFR_1, averageTFR_2


from .Bipolar import bipolar_epoch
from .csd import CSD_epoch


def compute_TFR(pid, selectives, condition1_preprocessing, condition2_preprocessing, freqs, TF_parameters , version = 'bipolar' ):

    # get the epochs for the BiasRight and BiasLeft conditions
    c1_epochs = get_epoch_StimOn(pid, **condition1_preprocessing)
    c2_epochs = get_epoch_StimOn(pid, **condition2_preprocessing)

    if version == 'bipolar':
        selectives = selectives[~selectives['ch_indexs'].isin([383])] # remove the last channel if it is selected
        c1_epochs = bipolar_epoch(c1_epochs)
        c2_epochs = bipolar_epoch(c2_epochs)
    if version == 'CSD':
        channels = get_channels(pid)
        selectives = selectives[~selectives['ch_indexs'].isin([0,383])] # remove the first and last channel if they are selected
        c1_epochs = CSD_epoch(c1_epochs, channels)
        c2_epochs = CSD_epoch(c2_epochs, channels)
    ch_index = selectives.loc[selectives['pid'] == pid, 'ch_indexs'].tolist()
    acronyms = selectives.loc[selectives['pid'] == pid, 'acronyms'].tolist()
    # select channels
    ch_names_c1 = [ch for i,ch in enumerate(c1_epochs.info['ch_names']) if i in ch_index]
    ch_names_c2 = [ch for i,ch in enumerate(c2_epochs.info['ch_names']) if i in ch_index]
    c1_epochs = c1_epochs.pick_channels(ch_names= ch_names_c1)
    c2_epochs = c2_epochs.pick_channels(ch_names= ch_names_c2)

    # compute TFR
    n_cycles = TF_parameters['n_cycles']
    time_bandwidth = TF_parameters['time_bandwidth']
    n_jobs = TF_parameters['n_jobs']

    c1_TFR = mne.time_frequency.tfr_multitaper(c1_epochs, freqs = freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, n_jobs = n_jobs, return_itc = False)
    c2_TFR = mne.time_frequency.tfr_multitaper(c2_epochs, freqs = freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, n_jobs = n_jobs, return_itc = False)


    c1 = np.log(c1_TFR.data)
    c2 = np.log(c2_TFR.data)
    diff = c1 - c2

    n_trials_c1 = c1_epochs._data.shape[0]
    n_trials_c2 = c2_epochs._data.shape[0]
    times = c1_TFR.times
    freqs = c1_TFR.freqs
    return diff, n_trials_c1, n_trials_c2, ch_index, acronyms,  freqs, times