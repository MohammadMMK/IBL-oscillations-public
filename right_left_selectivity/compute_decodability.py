
from firing_rate import right_left_firingRates_onDepths
import numpy as np
import pickle


def compute_decodability(z_score_condition1, z_score_condition2):
    """
    Compute the decodability between two conditions using AUROC (Area Under the Receiver Operating Characteristic curve).
    This function takes the z-scored firing rates of two conditions and computes the AUROC for each depth and time point.
    It uses logistic regression to fit the model and handles class imbalance by downsampling the majority class.
    Parameters:
    z_score_condition1 (numpy.ndarray): A 3D array of z-scored firing rates for condition 1 with shape (n_c1_trials, n_depths, n_time).
    z_score_condition2 (numpy.ndarray): A 3D array of z-scored firing rates for condition 2 with shape (n_c2_trials, n_depths, n_time).
    Returns:
    numpy.ndarray: A 2D array of AUROC values with shape (n_depths, n_time).
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import resample
    import numpy as np
    n_c1_trials, n_depths, n_time = z_score_condition1.shape
    n_c2_trials = z_score_condition2.shape[0]

    # Initialize a storage for AUROC values
    auroc_values = np.zeros((n_depths, n_time))

    for depth in range(n_depths):
        for time in range(n_time):
            # Prepare the feature set and labels
            firing_rates = np.concatenate((z_score_condition1[:, depth, time], z_score_condition2[:, depth, time]))
            labels = np.concatenate((np.ones(n_c1_trials), np.zeros(n_c2_trials)))  # 1 for condition 1, 0 for condition2

            # Handle class imbalance: downsample the majority class (right trials)   ##################### need improvement#####################
            if n_c1_trials > n_c2_trials:
                firing_rates, labels = resample(firing_rates, labels, 
                                                replace=False, 
                                                n_samples=n_c2_trials, 
                                                random_state=42)
            
            # Fit logistic regression
            model = LogisticRegression()
            model.fit(firing_rates.reshape(-1, 1), labels)  # Reshape for a single feature

            # Make predictions and calculate AUROC
            probabilities = model.predict_proba(firing_rates.reshape(-1, 1))[:, 1]
            auroc = roc_auc_score(labels, probabilities)
            
            # Store the AUROC value
            auroc_values[depth, time] = auroc

    return auroc_values

def Right_left_decodability_SpikesOnDepths(pids, eids, average_period = [0.1, 1], output_path = 'data/decodability_results.pkl', t_bin=0.1, d_bin=20, pre_stim=0.4, post_stim=1, depth_lim=[0, 3840]):
    """
    Computes the decodability of right vs left conditions based on spike data across different depths.
    Parameters:
    pids (list): List of probe IDs.
    eids (list): List of experiment IDs.
    average_period (list, optional): Time period over which to average the decodability scores. Default is [0.1, 1].
    output_path (str, optional): Path to save the results as a pickle file. Default is 'data/decodability_results.pkl'.
    t_bin (float, optional): Time bin size for spike data. Default is 0.1.
    d_bin (int, optional): Depth bin size for spike data. Default is 20.
    pre_stim (float, optional): Pre-stimulus time window. Default is 0.4.
    post_stim (float, optional): Post-stimulus time window. Default is 1.
    depth_lim (list, optional): Depth limits for spike data. Default is [0, 3840].
    Returns:
    dict: A dictionary containing the decodability scores, acronyms, and channel indices for each probe ID.
    The results are also saved to a pickle file specified by `output_path`.
    """

    results = {}
    for i, (pid, eid) in enumerate(zip(pids, eids), start=1):
        print(f'Processing pid {i} out of {len(pids)}')

        try:
            z_score_right, z_score_left,  times, depths, ids, acronyms, ch_indexs = right_left_firingRates_onDepths(eid, pid, t_bin=t_bin, d_bin=d_bin, pre_stim=pre_stim, post_stim=post_stim, depth_lim=depth_lim)
            decodability_scores = compute_decodability(z_score_condition1 = z_score_right, z_score_condition2 = z_score_left)
            print(f'Finished Decodability for pid {pid} and eid {eid}')

            # filter the decodability scores 
            idxs_time = np.where((times >= average_period[0]) & (times <= average_period[1]))[0]            
            decodability_scores_filtered = decodability_scores[:, idxs_time]
            # average the decodability scores over the time selected
            decodability_scores_filtered = np.mean(decodability_scores_filtered, axis=1, keepdims=False)

            results[pid] = {'decodability_scores': decodability_scores_filtered, 'acronyms': acronyms, 'ch_indexs': ch_indexs}

        except Exception as e:
            print(f'Error with pid {pid} and eid {eid}')
            print(e)
            continue
            
    # Save results to pickle file
    with open(output_path, 'wb') as pickle_file:
        pickle.dump(results, pickle_file)

    return results
