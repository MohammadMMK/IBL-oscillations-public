import os
import re
import mne
import numpy as np
import pandas as pd
from scipy.stats import skew

def epoch_stimOnset(raw, behavior,region, tmin=-0.5, tmax=1, baseline= None):
    bad_channels = list(raw.info['bads'])
    Visual_channels_clean = [ch for ch in raw.ch_names if isinstance(ch, str) and re.match(rf'^{region}[12456]', ch) and ch not in bad_channels]
    raw = raw.pick_channels(Visual_channels_clean)
    print(f' size visual channels clean: {len(Visual_channels_clean)}')
    print(f' size visual channels raw after channel picking: {len(raw.ch_names)}')
    print(Visual_channels_clean)
    print(f' visual clean channels: {Visual_channels_clean}')
    print(f"Bad channels : {raw.info['bads']}")
    print(f'visual channels raw after channel picking: {raw.ch_names}')
    
    events, event_id = mne.events_from_annotations(raw)
    behavior_stimOnset = behavior.dropna(subset=['stimOn_times']) 
    stimOn_times_events = events[events[:, 2] == event_id['stimOn_times']]
    epochs = mne.Epochs(raw, stimOn_times_events, event_id['stimOn_times'], tmin, tmax, baseline= baseline, preload=True, metadata=behavior_stimOnset)
    print(f' channels after epoching : {epochs.ch_names}')
    # compute_skewness
    keep_skewness=[]
    for i, epoch in enumerate(epochs):
        deriv=np.diff(epoch, axis=1)
        epoch_skewness=skew(np.abs(deriv),axis=1)
        keep_skewness.append(epoch_skewness)
    epochs.metadata['skewness'] = np.mean(keep_skewness,axis=1)
    return epochs

  
