import numpy as np
import mne
def numpyToMNE(epochs_full, behavior, ch_names, sfreq = 500   ):

    behavior = behavior.dropna(subset=['stimOn_times']).reset_index(drop=True)
    events = [(i, 0, 1) for i in range(len(behavior))]
    events = np.array(events)  # Convert events to numpy array
    if len(events) != len(behavior):
        print('Warning: events and behavior do not have the same length')
        print(f'events: {len(events)}, behavior: {len(behavior)}')
    info = mne.create_info(ch_names=ch_names,
                            sfreq=sfreq, ch_types='seeg')
    epochs_mne = mne.EpochsArray(epochs_full, info, events = events, event_id={'stimulus': 1}, tmin= -1, metadata = behavior)
    return epochs_mne
