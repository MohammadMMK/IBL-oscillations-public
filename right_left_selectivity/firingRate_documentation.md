# Firing Rate Computation

## Functions

### 1. `firingRates_onDepths`

Calculates Z-scored firing rates across depth bins for specified stimulus events.

- **Parameters**:
  - `stim_events` (dict): Dictionary with keys as stimulus types and values as arrays of stimulus onset times.
  - `spike_times` (np.ndarray): Array of spike times.
  - `spike_depths` (np.ndarray): Array of spike depths.
  - `t_bin` (float): Time bin size in seconds.
  - `d_bin` (int): Depth bin size in micrometers.
  - `pre_stim` (float): Time before stimulus onset (seconds).
  - `post_stim` (float): Time after stimulus onset (seconds).
  - `depth_lim` (list): Depth range for analysis in micrometers.
  
- **Returns**:
  - `z_scores` (dict): Z-scored firing rates per stimulus type, with shape `(num_trials, num_depths, num_time_bins)`.
  - `time_centers` (np.ndarray): Array of time bin centers, with shape `(num_time_bins,)`.
  - `depth_centers` (np.ndarray): Array of depth bin centers, with shape `(num_depths,)`.

### 2. `firing_Rates_onClusters`

Calculates Z-scored firing rates across cluster bins for specified stimulus events.

- **Parameters**:
  - `stim_events` (dict): Dictionary with keys as stimulus types and values as arrays of stimulus onset times.
  - `spike_times` (np.ndarray): Array of spike times.
  - `spike_clusters` (np.ndarray): Array of spike cluster IDs.
  - `t_bin` (float): Time bin size in seconds.
  - `pre_stim` (float): Time before stimulus onset (seconds).
  - `post_stim` (float): Time after stimulus onset (seconds).

- **Returns**:
  - `z_scores` (dict): Z-scored firing rates per stimulus type, with shape `(num_trials, num_clusters, num_time_bins)`.
  - `times` (np.ndarray): Array of time bin centers, with shape `(num_time_bins,)`.
  - `clusters` (np.ndarray): Array of unique cluster IDs, with shape `(num_clusters,)`.

### 3. `right_left_firingRates_onDepths`

Computes Z-scored firing rates for right and left stimulus events on depth bins.

- **Parameters**:
  - `eid` (str): Experiment ID.
  - `pid` (str): Probe insertion ID.
  - `t_bin` (float): Time bin size in seconds.
  - `d_bin` (int): Depth bin size in micrometers.
  - `pre_stim` (float): Time before stimulus onset (seconds).
  - `post_stim` (float): Time after stimulus onset (seconds).
  - `depth_lim` (list): Depth range for analysis in micrometers.
  - `modee` (str): Data loading mode, either `'download'` or `'load'` from local directory.

- **Returns**:
  - `z_score_right` (np.ndarray): Z-scored firing rates for right stimulus trials, with shape `(num_trials, num_depths, num_time_bins)`.
  - `z_score_left` (np.ndarray): Z-scored firing rates for left stimulus trials, with shape `(num_trials, num_depths, num_time_bins)`.
  - `times` (np.ndarray): Array of time bin centers, with shape `(num_time_bins,)`. (15)
  - `depths` (np.ndarray): Array of depth bin centers, with shape `(num_depths,)`. (192)
  - `ids` (np.ndarray): Array of unique channel IDs, with shape `(num_depths,)`.
  - `acronyms` (np.ndarray): Array of brain region acronyms for each depth, with shape `(num_depths,)`.
  - `ch_indexs` (np.ndarray): Array of channel indices, with shape `(num_depths,)`.
  - `coordinates` (np.ndarray): Array of 3D coordinates for each channel, with shape `(num_depths, 3)`.
  - `trial_indx` (np.ndarray): Array of trial indices for right and left events, with shape `(num_trials,)`.
  - `distance_to_change` (np.ndarray): Array of distances to the last probability change, with shape `(num_trials,)`.
  - `probs_left` (np.ndarray): Array of probability of left choices for each trial, with shape `(num_trials,)`.

## Usage
for right_left_firingRates_onDepths
```python
from firing_rate import firingRates_onDepths, firing_Rates_onClusters, right_left_firingRates_onDepths

# Example usage
z_score_right, z_score_left, times, depths, ids, acronyms, ch_indexs, coordinates, trial_indx, distance_to_change, probs_left = right_left_firingRates_onDepths(
    eid='6ab9d98c-b1e9-4574-b8fe-b9eec88097e0' ,
    pid= 'e4ce2e94-6fb9-4afe-acbf-6f5a3498602e',
    t_bin=0.1,
    d_bin=20,
    pre_stim=0.4,
    post_stim=1,
    depth_lim=[10, 3840], # always put it between 10 and 3840 in this case we will get the depths center from 20 to 3840 with 20 micrometers step exactly equal to depths of channels
    modee='download'
)
```
## Dependencies
IBL virtual enviroment to download and bin data 

