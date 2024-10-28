
import mne
import numpy as np
import pandas as pd
def new_annotation(raw, behavior):
    """
    Replace the existing annotations in a raw MNE object with new annotations based
    on behavioral event columns.

    Parameters:
    - raw: An instance of mne.io.Raw
    - behavior: A pandas DataFrame containing behavioral events with timestamps

    Returns:
    - The raw object with updated annotations.
    """
    # Validate input types (simple checks)
  
    if not isinstance(raw, mne.io.Raw):
        raise ValueError('raw must be an instance of mne.io.Raw')
    if not hasattr(behavior, 'loc'):
        raise ValueError('behavior must be a pandas DataFrame')
    

    # Remove old annotations
    raw.set_annotations(None)

    events_columns = [
        'goCueTrigger_times', 'stimOff_times', 'feedback_times',
        'firstMovement_times', 'response_times', 'stimOn_times'
    ]
    
    onsets, durations, descriptions = [], [], []

    # Collect new annotation data
    for column in events_columns:
        column_onsets = behavior[column].dropna().to_numpy()
        column_durations = np.zeros(len(column_onsets))
        column_descriptions = [column] * len(column_onsets)

        onsets.append(column_onsets)
        durations.append(column_durations)
        descriptions.extend(column_descriptions)
    
    onsets = np.concatenate(onsets)
    durations = np.concatenate(durations)
    start_time = raw.first_samp / raw.info['sfreq']
    
    # Create and set new annotations
    annotations = mne.Annotations(onset=onsets - start_time,
                                duration=durations,
                                description=descriptions,
                                orig_time=None)
    raw.set_annotations(annotations)
        

    return raw