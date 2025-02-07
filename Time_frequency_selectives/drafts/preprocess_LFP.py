

import os
from pathlib import Path
import sys
sys.path.append(str(Path(os.getcwd()).resolve().parent.parent)) # add the root of the project to the python path
from extraction_data import get_epoch_StimOn, get_behavior, get_channels, numpyToMNE
import numpy as np
import mne
from .csd import CSD_epoch
from .Bipolar import bipolar_epoch


def preprocess_LFP(pid, eid, ch_index, **kwargs):
    """
    Preprocess the LFP data for a given probe insertion ID (pid) and experiment ID (eid).
    
    Parameters:
    pid (str): Probe insertion ID.
    eid (str): Experiment ID.
    ch_index (list or str): List of channel indices to select or 'all' to select all channels.
    **kwargs: Additional keyword arguments for preprocessing options.
        - tmin (float): Start time of the epoch in seconds. Default is -0.5.
        - tmax (float): End time of the epoch in seconds. Default is 1.
        - cpus (int): Number of CPUs to use for filtering. Default is 1.
        - bandpass_filter (list): List containing low and high frequency for bandpass filter. Default is [None, None].
        - csd (bool): Whether to compute Current Source Density (CSD). Default is False.
        - bipolar (bool): Whether to compute bipolar signals. Default is False.
    
    Returns:
    mne.Epochs: The preprocessed MNE Epochs object.
    
    Raises:
    ValueError: If both csd and bipolar are set to True.
    """
    tmin = kwargs.get('tmin', -0.5)
    tmax = kwargs.get('tmax', 1)
    cpus = kwargs.get('cpus', 1)
    bandpass_filter = kwargs.get('bandpass_filter', [None, None])
    csd = kwargs.get('csd', False)
    bipolar = kwargs.get('bipolar', False)
    sfreq = 500

    # Load data
    epochs_full = get_epoch_StimOn(pid, modee='load', ephys_path=paths["LFP"])
    behavior = get_behavior(eid, modee='download')
    channels = get_channels(eid, pid, modee='download')

    # Create MNE epochs
    acronyms = channels['acronym'].tolist()
    ch_names = [f'{ch}_{i}' for i, ch in enumerate(acronyms)]
    epochs_mne = numpyToMNE(epochs_full, behavior, ch_names, sfreq=sfreq)

    # Select channels
    if ch_index != 'all':
        ch_names = [f'{ch}_{i}' for i, ch in enumerate(acronyms) if i in ch_index]
        epochs_mne = epochs_mne.pick_channels(ch_names=ch_names)

    # Filtering
    epochs_mne = epochs_mne.crop(tmin=tmin, tmax=tmax)
    if bandpass_filter[0] is not None and bandpass_filter[1] is not None:
        epochs_mne = epochs_mne.filter(l_freq=bandpass_filter[0], h_freq=bandpass_filter[1], n_jobs=cpus)
    

    # CSD or Bipolar
    if csd and bipolar:
        raise ValueError('CSD and bipolar cannot be both True')
    elif csd:
        epochs_mne = CSD_epoch(epochs_mne, channels)
    elif bipolar:
        epochs_mne = bipolar_epoch(epochs_mne)

    return epochs_mne

