def prepare_data(pid_eid_pair, average_period=[0.1, 1], base_path='/mnt/data/AdaptiveControl/IBLrawdata/classification/preprocess_data', 
                 t_bin=0.1, d_bin=20, pre_stim=0.4, post_stim=1, depth_lim=[10, 3840], overwrite=False):
    """
    Prepares a DataFrame for each probe (pid), containing firing rate and metadata for classification analysis.

    Parameters:
    - pid_eid_pair: tuple of (probe_id, experiment_id)
    - average_period: list, time interval for averaging firing rate [start, end]
    - base_path: str, path for saving output data
    - t_bin: float, time bin size for firing rate extraction
    - d_bin: float, depth bin size for firing rate extraction
    - pre_stim: float, pre-stimulus period for extraction
    - post_stim: float, post-stimulus period for extraction
    - depth_lim: list, depth limits for probe [min_depth, max_depth]
    - overwrite: bool, whether to overwrite existing files
    """

    # Import necessary libraries
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils import resample
    from sklearn.metrics import accuracy_score
    import sys
    from pathlib import Path
    import pandas as pd
    import statsmodels.formula.api as smf

    # Import custom module from parent directory
    parent_dir = Path.cwd().parent
    sys.path.insert(0, str(parent_dir))
    from firing_rate import right_left_firingRates_onDepths

    # Initialize identifiers
    pid = pid_eid_pair[0]
    eid = pid_eid_pair[1]
    output_path = f'{base_path}/{pid}.pkl'

    # Check if output file already exists and if overwriting is allowed
    if Path(output_path).exists() and not overwrite:
        print(f'File {output_path} already exists')
        return

    print(f'Processing {pid} {eid}')

    # Extract firing rates and metadata for the specified insertion
    z_score_right, z_score_left, times, depths, ids, acronyms, ch_indexs, coordinates, trial_indx, distance_to_change, probs_left = right_left_firingRates_onDepths(
        eid, pid, t_bin=t_bin, d_bin=d_bin, pre_stim=pre_stim, post_stim=post_stim, depth_lim=depth_lim)
    print('Finished extracting firing rates')

    # Compute time indices within the specified averaging period
    time_indices = np.where((times >= average_period[0]) & (times <= average_period[1]))[0]

    # Average firing rates over the defined time window
    z_score_right_average = np.mean(z_score_right[:, :, time_indices], axis=2)
    z_score_left_average = np.mean(z_score_left[:, :, time_indices], axis=2)

    # Labels for right (1) and left (0) trials
    labels_c1 = np.ones(z_score_right.shape[0])
    labels_c2 = np.zeros(z_score_left.shape[0])
    labels = np.concatenate((labels_c1, labels_c2))

    # Stack firing rates data along trial dimension
    firing_rates = np.concatenate((z_score_right_average, z_score_left_average), axis=0)

    # Expand metadata to match firing rates data shape
    x_coords = np.tile(coordinates[:, 0], firing_rates.shape[0])
    y_coords = np.tile(coordinates[:, 1], firing_rates.shape[0])
    z_coords = np.tile(coordinates[:, 2], firing_rates.shape[0])
    acronyms_repeated = np.tile(acronyms, firing_rates.shape[0])
    ch_indexs_repeated = np.tile(ch_indexs, firing_rates.shape[0])
    labels_repeated = np.tile(labels, firing_rates.shape[1])
    trial_indx_repeated = np.tile(trial_indx, firing_rates.shape[1])
    distance_to_change_repeated = np.tile(distance_to_change, firing_rates.shape[1])
    probs_left_repeated = np.tile(probs_left, firing_rates.shape[1])

    # Repeat probe and experiment IDs to match data structure
    probe_ids = np.repeat(pid, firing_rates.size)
    experiment_ids = np.repeat(eid, firing_rates.size)
    try:
        # Create DataFrame with structured data
        df = pd.DataFrame({
            'firing_rate': firing_rates.flatten(),
            'label': labels_repeated,
            'trial_index': trial_indx_repeated,
            'distance_to_change': distance_to_change_repeated,
            'prob_left': probs_left_repeated,
            'x': x_coords,
            'y': y_coords,
            'z': z_coords,
            'ch_index': ch_indexs_repeated,
            'acronym': acronyms_repeated,
            'probe_id': probe_ids,
            'experiment_id': experiment_ids
        })
    except ValueError:
        print('Error: Mismatch in data structure')
        # Check shapes of original arrays before flattening/repeating
        print("Original array shapes:")
        print(f"firing_rates: {firing_rates.shape}")
        print(f"labels_repeated: {labels_repeated.shape}")
        print(f"trial_indx_repeated: {trial_indx_repeated.shape}")
        print(f"distance_to_change_repeated: {distance_to_change_repeated.shape}")
        print(f"probs_left_repeated: {probs_left_repeated.shape}")
        print(f"x_coords: {x_coords.shape}")
        print(f"y_coords: {y_coords.shape}")
        print(f"z_coords: {z_coords.shape}")
        print(f"ch_indexs_repeated: {ch_indexs_repeated.shape}")
        print(f"acronyms_repeated: {len(acronyms_repeated)}")  # if it's a list
        print(f"probe_ids: {len(probe_ids)}")  # if it's a list
        print(f"experiment_ids: {len(experiment_ids)}")  # if it's a list
        return 1
   

    # Save the prepared DataFrame
    df.to_pickle(output_path)
    print(f'Data saved to {output_path}')
    return 0
