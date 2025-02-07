
import os
from pathlib import Path
from extraction_data import get_epoch_StimOn, get_channels
import numpy as np
import mne
import pickle
from .Bipolar import bipolar_epoch
from .csd import CSD_epoch


def compute_TFR(pid, ch_index, condition1_preprocessing, condition2_preprocessing, freqs, TF_parameters, version='bipolar'):
    """
    Compute the Time-Frequency Representation (TFR) for two conditions and return the difference.
    Parameters:
    pid : str
        Participant ID.
    ch_index : list of int
        List of channel indices to select.
    condition1_preprocessing : dict
        Preprocessing parameters for the first condition.
    condition2_preprocessing : dict
        Preprocessing parameters for the second condition.
    freqs : array-like
        Array of frequencies of interest.
    TF_parameters : dict
        Parameters for TFR computation, including:
        - 'n_cycles': Number of cycles in the wavelet.
        - 'time_bandwidth': Time bandwidth product.
        - 'n_jobs': Number of jobs to run in parallel.
    version : str, optional
        Version of the data to use ('bipolar' or 'CSD'), by default 'bipolar'.
    Returns:
    tuple
        A tuple containing:
        - diff (ndarray): Log-transformed difference between the TFRs of the two conditions.
        - n_trials_c1 (int): Number of trials for the first condition.
        - n_trials_c2 (int): Number of trials for the second condition.
        - freqs (ndarray): Array of frequencies.
        - times (ndarray): Array of time points.
    """

    # get the epochs for the BiasRight and BiasLeft conditions
    c1_epochs = get_epoch_StimOn(pid, **condition1_preprocessing)
    c2_epochs = get_epoch_StimOn(pid, **condition2_preprocessing)

    if version == 'bipolar':
        c1_epochs = bipolar_epoch(c1_epochs)
        c2_epochs = bipolar_epoch(c2_epochs)
    if version == 'CSD':
        channels = get_channels(pid)
        c1_epochs = CSD_epoch(c1_epochs, channels)
        c2_epochs = CSD_epoch(c2_epochs, channels)

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
    return diff, n_trials_c1, n_trials_c2,  freqs, times


def TF_in_one_big_job( decoding_result, c1_preprocessing, c2_preprocessing, freqs, TF_parameters, version, file_name):
    """
    Compute time-frequency representations (TFR) for each channel and save the results.
    Parameters:
    -----------
    decoding_result : pandas.DataFrame
        DataFrame containing decoding results with columns 'pid', 'ch_indexs', 'accuracies_c1', 'accuracies_c2', 'p_value_c1', and 'p_value_c2'.
    c1_preprocessing : dict
        Preprocessing parameters for condition 1.
    c2_preprocessing : dict
        Preprocessing parameters for condition 2.
    freqs : array-like
        Array of frequencies for TFR computation.
    TF_parameters : dict
        Parameters for TFR computation.
    version : str
        Version of the analysis ('bipolar' or 'CSD').
    file_name : str
        Name of the file to save the results.
    Returns:
    --------
    None
    Saves:
    ------
    A pickle file containing a list of dictionaries, each dictionary contains:
        - 'TF': Time-frequency representation (2D array:  frequencies x times)
        - 'pid': Participant ID
        - 'n_trials_c1': Number of trials for condition 1
        - 'n_trials_c2': Number of trials for condition 2
        - 'ch_index': Channel index
        - 'acronym': Brain region acronym
        - 'accuracy_right': Decoding accuracy for right stim
        - 'accuracy_left': Decoding accuracy for left stim
        - 'pvalue_right': p-value for right stim
        - 'pvalue_left': p-value for left stim
        - 'freqs': Frequencies used for TFR computation
        - 'times': Time points used for TFR computation
    """


    pids = decoding_result['pid'].unique()
    results = []
    for i, pid in enumerate(pids):

        print(f'Processing pid {pid} ({i + 1}/{len(pids)})')

        # get list of channels from decoding_result
        if version == 'bipolar':
            decoding_df = decoding_df[~decoding_df['ch_indexs'].isin([383])] # remove the last channel if it is selected
        if version == 'CSD':
            decoding_df = decoding_df[~decoding_df['ch_indexs'].isin([0,383])] # remove the first and last channel if they are selected

        ch_index = decoding_df.loc[decoding_df['pid'] == pid, 'ch_indexs'].tolist()
        acronyms = decoding_df.loc[decoding_df['pid'] == pid, 'acronyms'].tolist()


        diff,  n_trials_c1, n_trials_c2,  frequencies, times = compute_TFR(pid, ch_index, c1_preprocessing, c2_preprocessing, freqs, TF_parameters, version = version)
        for i, ch in enumerate(ch_index):

            accuracy_right = decoding_result.loc[(decoding_result['pid'] == pid) & (decoding_result['ch_indexs'] == ch), 'accuracies_c1'].values[0]   
            accuracy_left = decoding_result.loc[(decoding_result['pid'] == pid) & (decoding_result['ch_indexs'] == ch), 'accuracies_c2'].values[0]
            pvalue_right = decoding_result.loc[(decoding_result['pid'] == pid) & (decoding_result['ch_indexs'] == ch), 'p_value_c1'].values[0]
            pvalue_left = decoding_result.loc[(decoding_result['pid'] == pid) & (decoding_result['ch_indexs'] == ch), 'p_value_c2'].values[0]
             

            data= {'TF': diff[i, : , :] ,'pid': pid, 'n_trials_c1': n_trials_c1, 'n_trials_c2': n_trials_c2, 'ch_index': ch, 'acronym': acronyms[i],
                   'accuracy_right': accuracy_right, 'accuracy_left':accuracy_left, 'pvalue_right': pvalue_right,'pvalue_left':pvalue_left,  'freqs': frequencies, 'times': times}
            
            results.append(data)
    # save the results
    results_dir = os.path.join(str(Path(os.getcwd()).resolve().parent), 'TF_data')
    os.makedirs(results_dir, exist_ok = True)
    file_name = os.path.join(results_dir, file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(results, f)



import pickle
def load_tf_data(file_path):
    """
    Load TF data from a pickle file.

    Parameters
    ----------
    file_path : str
        Path to the pickle file.

    Returns
    -------
    data : list of dict
        List of dictionaries containing the TF data for each channel.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data