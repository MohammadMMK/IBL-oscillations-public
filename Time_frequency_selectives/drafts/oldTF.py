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