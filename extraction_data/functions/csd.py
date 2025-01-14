# CSD
import numpy as np
def CSD_epoch(epochs_mne_filter, channels):
    epochs_full = epochs_mne_filter.get_data()
    ch_coords = channels[['axial_um', 'lateral_um']].to_numpy()
    ch_dist = np.sqrt(np.sum(np.diff(ch_coords, axis=0)**2, axis=1))
    # Compute differences along the channel axis (axis=1)
    v_diff1 = np.diff(epochs_full[:, :-1, :], axis=1)  # Exclude the last channel
    v_diff2 = np.diff(epochs_full[:, 1:, :], axis=1)   # Exclude the first channel
    # Expand the channel distance matrix to match the new 3D shape
    ch_dist_mat_expanded = np.tile(ch_dist[:, np.newaxis, np.newaxis], (1, epochs_full.shape[2], epochs_full.shape[0]))
    ch_dist_mat_expanded = np.transpose(ch_dist_mat_expanded, (2, 0, 1))  # Match dimensions: trials, channels, time
    # Compute the CSD
    csd = (v_diff2 / ch_dist_mat_expanded[:, 1:, :]) - (v_diff1 / ch_dist_mat_expanded[:, :-1, :])
    padded_epochs = np.pad(csd, ((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=0)
    epochs_mne_filter._data = padded_epochs
    return epochs_mne_filter

