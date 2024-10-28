import numpy as np
import os
import sys
import mne
import pandas as pd


def TFR_diff( pid, eid, **kwargs):
 
    """
    Compute the time-frequency representation (TFR) difference between two conditions for a given probe and experiment.
    Parameters:
    -----------
    pid : str
        probe ID.
    eid : str
        Experiment ID.
    **kwargs : dict
        Additional keyword arguments:
        - condition (str): The condition to compare. Possible values are 'Stim_NoStim', 'Right_Left', 'BiasRight_BiasLeft', 
            'success_error', 'PrevSuccess_PrevFail', 'expected_unexpected_stim', 'Right_left_choice'.
        - n_jobs (int): Number of jobs to run in parallel.
        - region (str): Brain region of interest.
        - tmin (float): Start time before event.
        - tmax (float): End time after event.
        - remove_first_trials_of_block (bool): Whether to remove the first trials of each block.
        - overwrite (bool): Whether to overwrite existing files.
        - min_trial (int): Minimum number of trials required for each condition.
    Returns:
    --------
    None
        The function saves the TFR difference to a file and does not return any value.
    Raises:
    -------
    ValueError
        If an invalid condition is provided.
    """

    condition = kwargs.get('condition')
    n_jobs = kwargs.get('n_jobs')
    region = kwargs.get('region')
    tmin = kwargs.get('tmin')
    tmax = kwargs.get('tmax')
    remove_first_trials_of_block = kwargs.get('remove_first_trials_of_block')
    overwrite = kwargs.get('overwrite')
    min_trial = kwargs.get('min_trial')


    sys.path.append(os.path.abspath('/crnldata/cophy/TeamProjects/mohammad/ibl-oscillations/_analyses/TFR/functions'))
    from new_annotation import new_annotation
    from epoching import epoch_stimOnset
    
    freqs = np.concatenate([np.arange(1, 10, 0.5), np.arange(10, 45, 1)])
    n_cycles = freqs / 2.
    time_bandwidth = 3.5
    path_save = f'/mnt/data/AdaptiveControl/IBLrawdata/TF_data/{region}/TFR_{condition}_{pid}.npy'

    if os.path.isfile(path_save) and overwrite == False:
        print(f'File already exists for pid: {pid}')
        return 
    
    os.makedirs(f'/mnt/data/AdaptiveControl/IBLrawdata/TF_data/{region}/', exist_ok=True)

    ######################
    # Load data
    ########################
    path_behavior = f'/mnt/data/AdaptiveControl/IBLrawdata/eid_data/{eid}/trials_{eid}.pkl'
    behavior = pd.read_pickle(path_behavior)
    path_lfp = f'/mnt/data/AdaptiveControl/IBLrawdata/pid_data/{pid}/lfp_{pid}_raw.fif'
    lfp = mne.io.read_raw_fif(path_lfp, preload=True)
    ######################
    # epoching
    #######################
    lfp = new_annotation(lfp, behavior)
    epoch = epoch_stimOnset(lfp, behavior,region,  tmin=tmin, tmax=tmax) # remove bad_channels, and none intrested area channels, and add skewness value for epoch into metadata
    # rwmove epochs with skewness > 1.5
    clean = epoch.metadata[epoch.metadata['skewness'] < 1.5].index
    epoch = epoch[clean]

    ######################
    # re-refrence
    #######################
    data = epoch.get_data()
    mean_across_channels = np.mean(data, axis=1, keepdims=True)
    data_centered = data - mean_across_channels
    epoch._data = data_centered


    ######################
    # get epoch indices for conditions
    #######################
    meta = epoch.metadata.reset_index(drop=True)

    if condition == 'Stim_NoStim':
        condition1_trial = np.where(((meta['contrastLeft'] ==1) | (meta['contrastRight'] == 1)))[0]
        condition2_trial = np.where(((meta['contrastLeft'] <0.1) | (meta['contrastRight'] < 0.1)))[0]
    elif condition == 'Right_Left':
        condition1_trial = np.where(((meta['contrastRight'].isna()) & (meta['contrastLeft'] > 0)))[0]
        condition2_trial = np.where(((meta['contrastLeft'].isna()) & (meta['contrastRight'] > 0)))[0]
    elif condition == 'BiasRight_BiasLeft':
        condition1_trial = np.where(meta['probabilityLeft'] == 0.2 )[0]
        condition2_trial = np.where(meta['probabilityLeft'] == 0.8 )[0]
    elif condition == 'success_error':
        condition1_trial = np.where(meta['feedbackType'] == 1)[0]
        condition2_trial = np.where(meta['feedbackType'] == -1)[0]
    elif condition == 'PrevSuccess_PrevFail':
        condition1_trial = np.where(meta['feedbackType'].shift(1) == 1)[0]
        condition2_trial = np.where(meta['feedbackType'].shift(1) == -1)[0]  
    elif condition == 'expected_unexpected_stim':
        condition1_trial = np.where(((meta['probabilityLeft'] == 0.8) & (meta['contrastLeft'] > 0.2)) |
                                    ((meta['probabilityLeft'] == 0.2) & (meta['contrastRight'] > 0.2)))[0]
        condition2_trial = np.where(((meta['probabilityLeft'] == 0.8) & (meta['contrastRight'] > 0.2)) |
                                    ((meta['probabilityLeft'] == 0.2) & (meta['contrastLeft'] > 0.2)))[0]
    elif condition == 'expected_unexpected_NoStim':
        condition1_trial = np.where(((meta['probabilityLeft'] == 0.8) & (meta['contrastLeft'] < 0.1)) |
                                    ((meta['probabilityLeft'] == 0.2) & (meta['contrastRight'] < 0.1)))[0]
        condition2_trial = np.where(((meta['probabilityLeft'] == 0.8) & (meta['contrastRight'] < 0.1)) |
                                    ((meta['probabilityLeft'] == 0.2) & (meta['contrastLeft'] < 0.1)))[0]
    elif condition == 'Right_left_choice':
        condition1_trial = np.where(meta['choice'] == 1)[0]
        condition2_trial = np.where(meta['choice'] == -1)[0]
    else:
        raise ValueError('Invalid condition')

    
    change_indices = meta['probabilityLeft'].ne(meta['probabilityLeft'].shift()).to_numpy().nonzero()[0]
    change_indices =  change_indices[1:] # remove the first change
    change_indices_10 = []
    for change in change_indices:
        change_indices_10.extend(range(change, change + 11))


    if remove_first_trials_of_block:
        condition1_trial = [i for i in condition1_trial if i not in change_indices_10]
        condition2_trial = [i for i in condition2_trial if i not in change_indices_10]

    if len(condition1_trial) < min_trial or len(condition2_trial) < min_trial:
        print(f'Number of epochs for  is less than {min_trial} for pid: {pid}')
        return 0 
    epochs_1 = epoch[condition1_trial]
    epochs_2 = epoch[condition2_trial]

    # ######################
    # # compute TFR difference
    # #######################
    averageTFR_1 = mne.time_frequency.tfr_multitaper(epochs_1, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, return_itc=False, average=True, n_jobs=n_jobs)
    averageTFR_2 = mne.time_frequency.tfr_multitaper(epochs_2, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, return_itc=False, average=True, n_jobs=n_jobs)
    log1 = np.log(averageTFR_1.data)
    log2 = np.log(averageTFR_2.data)
    diff = log1 - log2
    np.save(path_save, diff)
    return 
