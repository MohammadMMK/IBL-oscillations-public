
from firing_rate import right_left_firingRates_onDepths
import numpy as np
import pickle


### logistic_regression_decodability.py ###
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample


def compute_decodability_LR(z_score_condition1, z_score_condition2, k_folds=5):
    """
    Compute the decodability between two conditions using Logistic Regression.
    This function takes the z-scored firing rates of two conditions and computes the decodability for each depth and time point.
    The function uses cross-validation to ensure robust evaluation of classification performance.
    Parameters:
    z_score_condition1 (numpy.ndarray): A 3D array of z-scored firing rates for condition 1 with shape (n_c1_trials, n_depths, n_time).
    z_score_condition2 (numpy.ndarray): A 3D array of z-scored firing rates for condition 2 with shape (n_c2_trials, n_depths, n_time).
    k_folds (int): Number of folds for K-Fold cross-validation. Default is 5.

    Returns:
    numpy.ndarray: A 2D array of decodability scores with shape (n_depths, n_time).
    """
    n_c1_trials, n_depths, n_time = z_score_condition1.shape
    n_c2_trials = z_score_condition2.shape[0]

    # Initialize storage for decodability scores
    decodability_scores = np.zeros((n_depths, n_time))

    # Labels for conditions
    labels_c1 = np.ones(n_c1_trials)
    labels_c2 = np.zeros(n_c2_trials)

    for depth in range(n_depths):
        for time in range(n_time):
            # Prepare feature set and labels
            firing_rates = np.concatenate((z_score_condition1[:, depth, time], z_score_condition2[:, depth, time]))
            labels = np.concatenate((labels_c1, labels_c2))

            # Handle class imbalance using resampling if needed
            if len(labels_c1) > len(labels_c2):
                firing_rates, labels = resample(firing_rates, labels, replace=False, n_samples=len(labels_c2), random_state=42)

            # Standardize features
            scaler = StandardScaler()
            firing_rates_scaled = scaler.fit_transform(firing_rates.reshape(-1, 1))

            # Initialize cross-validation
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_auroc = []

            for train_index, test_index in skf.split(firing_rates_scaled, labels):
                X_train, X_test = firing_rates_scaled[train_index], firing_rates_scaled[test_index]
                y_train, y_test = labels[train_index], labels[test_index]

                # Fit Logistic Regression
                model = LogisticRegression(solver='lbfgs', max_iter=1000)
                model.fit(X_train, y_train)

                # Predict probabilities and calculate AUROC
                probabilities = model.predict_proba(X_test)[:, 1]
                auroc = roc_auc_score(y_test, probabilities)
                fold_auroc.append(auroc)

            # Store the average AUROC across folds
            decodability_scores[depth, time] = np.mean(fold_auroc)

    return decodability_scores


# def Right_left_decodability_SpikesOnDepths(pids, eids, average_period = [0.1, 1], output_path = 'data/decodability_results.pkl', t_bin=0.1, d_bin=20, pre_stim=0.4, post_stim=1, depth_lim=[0, 3840]):
#     """
#     Computes the decodability of right vs left conditions based on spike data across different depths.
#     Parameters:
#     pids (list): List of probe IDs.
#     eids (list): List of experiment IDs.
#     average_period (list, optional): Time period over which to average the decodability scores. Default is [0.1, 1].
#     output_path (str, optional): Path to save the results as a pickle file. Default is 'data/decodability_results.pkl'.
#     t_bin (float, optional): Time bin size for spike data. Default is 0.1.
#     d_bin (int, optional): Depth bin size for spike data. Default is 20.
#     pre_stim (float, optional): Pre-stimulus time window. Default is 0.4.
#     post_stim (float, optional): Post-stimulus time window. Default is 1.
#     depth_lim (list, optional): Depth limits for spike data. Default is [0, 3840].
#     Returns:
#     dict: A dictionary containing the decodability scores, acronyms, and channel indices for each probe ID.
#     The results are also saved to a pickle file specified by `output_path`.
#     """

#     results = {}
#     for i, (pid, eid) in enumerate(zip(pids, eids), start=1):
#         print(f'Processing pid {i} out of {len(pids)}')

#         try:
#             z_score_right, z_score_left,  times, depths, ids, acronyms, ch_indexs = right_left_firingRates_onDepths(eid, pid, t_bin=t_bin, d_bin=d_bin, pre_stim=pre_stim, post_stim=post_stim, depth_lim=depth_lim)
#             decodability_scores = compute_decodability_LR(z_score_condition1 = z_score_right, z_score_condition2 = z_score_left)
#             print(f'Finished Decodability for pid {pid} and eid {eid}')

#             # filter the decodability scores 
#             idxs_time = np.where((times >= average_period[0]) & (times <= average_period[1]))[0]            
#             decodability_scores_filtered = decodability_scores[:, idxs_time]
#             # average the decodability scores over the time selected
#             decodability_scores_filtered = np.mean(decodability_scores_filtered, axis=1, keepdims=False)

#             results[pid] = {'decodability_scores': decodability_scores_filtered, 'acronyms': acronyms, 'ch_indexs': ch_indexs}

#         except Exception as e:
#             print(f'Error with pid {pid} and eid {eid}')
#             print(e)
#             continue
            
#     # Save results to pickle file
#     with open(output_path, 'wb') as pickle_file:
#         pickle.dump(results, pickle_file)

#     return results
