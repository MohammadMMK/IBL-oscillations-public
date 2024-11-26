import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

def compute_decodability_LR(firingRates, labels, k_folds=5, bin_size=1):
    """
    Compute the decodability of neural firing rates using Logistic Regression with cross-validation.
    Parameters:
    firingRates (numpy.ndarray): A 3D array of shape (n_trials, n_depths, n_time) containing the firing rates.
    labels (numpy.ndarray): A 1D array of shape (n_trials,) containing the class labels for each trial.
    k_folds (int, optional): The number of folds for cross-validation. Default is 5.
    bin_size (int, optional): The number of consecutive time bins to combine. Default is 4.
    Returns:
    tuple: A tuple containing:
        - decodability_scores (numpy.ndarray): A 2D array of shape (n_depths, n_new_time) containing the average AUROC scores.
        - decodability_std (numpy.ndarray): A 2D array of shape (n_depths, n_new_time) containing the standard deviation of AUROC scores.
    """
    # Combine time bins by summing each `bin_size` consecutive bins
    n_trials, n_depths, n_time = firingRates.shape
    n_new_time = n_time // bin_size  # Calculate new number of time points after binning

    # Sum every `bin_size` bins together
    firingRates_binned = firingRates[:, :, :n_new_time * bin_size].reshape(
        n_trials, n_depths, n_new_time, bin_size
    ).sum(axis=-1)

    # Initialize storage for decodability scores
    decodability_scores = np.zeros((n_depths, n_new_time))
    decodability_std = np.zeros((n_depths, n_new_time))

    for depth in range(n_depths):
        for time in range(n_new_time):
            # Extract firing rates for the current depth and time point
            firing_rates = firingRates_binned[:, depth, time]

            # Reshape firing rates and labels to use with RandomUnderSampler
            firing_rates_reshaped = firing_rates.reshape(-1, 1)
            labels_reshaped = labels

            # Handle class imbalance using RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            firing_rates_balanced, labels_balanced = rus.fit_resample(firing_rates_reshaped, labels_reshaped)

            # Standardize features
            scaler = StandardScaler()
            firing_rates_scaled = scaler.fit_transform(firing_rates_balanced)

            # Initialize cross-validation
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_auroc = []

            for train_index, test_index in skf.split(firing_rates_scaled, labels_balanced):
                X_train, X_test = firing_rates_scaled[train_index], firing_rates_scaled[test_index]
                y_train, y_test = labels_balanced[train_index], labels_balanced[test_index]

                # Fit Logistic Regression
                model = LogisticRegression(solver='lbfgs', max_iter=1000)
                model.fit(X_train, y_train)

                # Predict probabilities and calculate AUROC
                predictions = model.predict(X_test)
                auroc = roc_auc_score(y_test, predictions, average=None)
                fold_auroc.append(auroc)

            # Store the average AUROC and standard deviation across folds
            decodability_scores[depth, time] = np.mean(fold_auroc)
            decodability_std[depth, time] = np.std(fold_auroc)

    return decodability_scores, decodability_std



def right_left_decodability(pid_eid_pair):
    import os 
    import pickle
    import pandas as pd

    base_path = '/mnt/data/AdaptiveControl/IBLrawdata/classification/preprocess_firingRate'
    pid = pid_eid_pair[0]
    path = os.path.join(base_path, f'{pid}.pkl')
    if not os.path.exists(path):
        print(f"File {path} not found")
        return 1

    with open(path, 'rb') as f:
        data = pickle.load(f)
    firing_rates = data['firing_rates'] # shape (n_trials, n_neurons, n_time_bins)
    trial_info = data['trial_info'] # dictionary keys ['trial_index', 'labels', 'contrasts', 'distance_to_change', 'prob_left', 'probe_id', 'experiment_id']
    depth_info = data['depth_info'] # dictionary keys ['depth', 'ids', 'acronyms', 'ch_indexs', 'x_coordinates', 'y_coordinates', 'z_coordinates']
    time_bins = data['time_bins'] # shape (n_time_bins,)

    # Filter conditions
    contrast_filter = trial_info['contrasts'] == 1
    # probability_left_filter = (trial_info['prob_left'] == 0.5) | np.isnan(trial_info['prob_left'])
    probability_left_filter = (trial_info['prob_left'] >= 0) 

    trial_info_filter = contrast_filter & probability_left_filter
    time_filter = time_bins > 0  # time is in milliseconds
    
    # Apply filters
    trial_info_filtered_dict = {key: value[trial_info_filter] for key, value in trial_info.items()}
    trial_info_filtered = pd.DataFrame(trial_info_filtered_dict)
    time_bins_filtered = time_bins[time_filter]
    firing_rates_filtered = firing_rates[trial_info_filter, :, :][:, :, time_filter]

    labels_filter = trial_info_filtered['labels']
    from compute_decodability import compute_decodability_LR
    decodability_scores, decodability_std = compute_decodability_LR(firing_rates_filtered, labels_filter)
    data['decodability_scores_active'] = decodability_scores
    data['decodability_std_active'] = decodability_std
    # save the data
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"File {path} saved")
    return 0