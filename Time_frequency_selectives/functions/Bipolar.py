import numpy as np
import mne

def bipolar_epoch(epochs_mne):
    """
    Compute the bipolar version of the epoch LFP.
    
    Parameters:
    epochs_mne (mne.Epochs): The MNE Epochs object containing the epochs.
    
    Returns:
    mne.Epochs: The MNE Epochs object with bipolar signals.
    """
    # Extract the data and channel names
    epochs_full = epochs_mne.get_data()
    ch_names = epochs_mne.ch_names
    
    # Generate new channel names for bipolar signals
    new_ch_names = [f"{ch1}-{ch2}" for ch1, ch2 in zip(ch_names[:-1], ch_names[1:])]
    
    # Compute bipolar signals by taking the difference between adjacent channels
    bipolar_signals = np.diff(epochs_full, axis=1)
    
    # Create a new MNE Epochs object with the bipolar signals
    info = mne.create_info(ch_names=new_ch_names, sfreq=epochs_mne.info['sfreq'], ch_types='eeg')
    bipolar_epochs = mne.EpochsArray(bipolar_signals, info, events=epochs_mne.events, tmin=epochs_mne.tmin)
    
    return bipolar_epochs