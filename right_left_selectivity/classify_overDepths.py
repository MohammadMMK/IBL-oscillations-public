
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import accuracy_score


def classify_depths_LR(z_score_condition1, z_score_condition2, times, k_folds=5, time_window=(0.1, 1.0)):
    """
    Classify each depth as right-selective, left-selective, or neutral based on firing rates in a given time window.
    This function takes the z-scored firing rates of two conditions and classifies each depth using Logistic Regression.
    The function uses cross-validation to ensure robust evaluation of classification performance.

    Parameters:
    z_score_condition1 (numpy.ndarray): A 3D array of z-scored firing rates for condition 1 with shape (n_c1_trials, n_depths, n_time).
    z_score_condition2 (numpy.ndarray): A 3D array of z-scored firing rates for condition 2 with shape (n_c2_trials, n_depths, n_time).
    times (numpy.ndarray): A 1D array of time points corresponding to the third dimension of z-scored firing rates.
    k_folds (int): Number of folds for K-Fold cross-validation. Default is 5.
    time_window (tuple): The time window (start, end) over which to analyze the firing rates. Default is (0.1, 1.0).

    Returns:
    dict: A dictionary with depth indices as keys and classification labels ('right-selective', 'left-selective', 'neutral') as values.
    """
    n_c1_trials, n_depths, n_time = z_score_condition1.shape
    n_c2_trials = z_score_condition2.shape[0]

    # Find the indices corresponding to the specified time window
    time_indices = np.where((times >= time_window[0]) & (times <= time_window[1]))[0]

    # Initialize storage for classification results
    depth_classification = {}

    # Labels for conditions
    labels_c1 = np.ones(n_c1_trials)
    labels_c2 = np.zeros(n_c2_trials)

    for depth in range(n_depths):
        # Prepare feature set and labels by averaging over the specified time window
        firing_rates_c1 = np.mean(z_score_condition1[:, depth, time_indices], axis=1)
        firing_rates_c2 = np.mean(z_score_condition2[:, depth, time_indices], axis=1)
        firing_rates = np.concatenate((firing_rates_c1, firing_rates_c2))
        labels = np.concatenate((labels_c1, labels_c2))

        # Handle class imbalance using resampling if needed
        if len(labels_c1) > len(labels_c2):
            firing_rates, labels = resample(firing_rates, labels, replace=False, n_samples=len(labels_c2), random_state=42)

        # Standardize features
        scaler = StandardScaler()
        firing_rates_scaled = scaler.fit_transform(firing_rates.reshape(-1, 1))

        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_accuracies = []

        for train_index, test_index in skf.split(firing_rates_scaled, labels):
            X_train, X_test = firing_rates_scaled[train_index], firing_rates_scaled[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Fit Logistic Regression
            model = LogisticRegression(solver='lbfgs', max_iter=1000)
            model.fit(X_train, y_train)

            # Predict labels and calculate accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)

        # Determine classification label based on average accuracy across folds
        avg_accuracy = np.mean(fold_accuracies)
        if avg_accuracy > 0.6:  # Threshold for selectivity
            depth_classification[depth] = 'right-selective' if np.mean(firing_rates_c1) > np.mean(firing_rates_c2) else 'left-selective'
        else:
            depth_classification[depth] = 'neutral'

    return depth_classification

def classify_depths_RF(z_score_condition1, z_score_condition2, times, k_folds=5, time_window=(0.1, 1.0)):
    """
    Classify each depth as right-selective, left-selective, or neutral based on firing rates in a given time window.
    This function takes the z-scored firing rates of two conditions and classifies each depth using a Random Forest Classifier.
    The function uses cross-validation to ensure robust evaluation of classification performance.

    Parameters:
    z_score_condition1 (numpy.ndarray): A 3D array of z-scored firing rates for condition 1 with shape (n_c1_trials, n_depths, n_time).
    z_score_condition2 (numpy.ndarray): A 3D array of z-scored firing rates for condition 2 with shape (n_c2_trials, n_depths, n_time).
    times (numpy.ndarray): A 1D array of time points corresponding to the third dimension of z-scored firing rates.
    k_folds (int): Number of folds for K-Fold cross-validation. Default is 5.
    time_window (tuple): The time window (start, end) over which to analyze the firing rates. Default is (0.1, 1.0).

    Returns:
    dict: A dictionary with depth indices as keys and classification labels ('right-selective', 'left-selective', 'neutral') as values.
    """
    n_c1_trials, n_depths, n_time = z_score_condition1.shape
    n_c2_trials = z_score_condition2.shape[0]

    # Find the indices corresponding to the specified time window
    time_indices = np.where((times >= time_window[0]) & (times <= time_window[1]))[0]

    # Initialize storage for classification results
    depth_classification = {}

    # Labels for conditions
    labels_c1 = np.ones(n_c1_trials)
    labels_c2 = np.zeros(n_c2_trials)

    for depth in range(n_depths):
        # Prepare feature set and labels by averaging over the specified time window
        firing_rates_c1 = np.mean(z_score_condition1[:, depth, time_indices], axis=1)
        firing_rates_c2 = np.mean(z_score_condition2[:, depth, time_indices], axis=1)
        firing_rates = np.concatenate((firing_rates_c1, firing_rates_c2))
        labels = np.concatenate((labels_c1, labels_c2))

        # Handle class imbalance using resampling if needed
        if len(labels_c1) > len(labels_c2):
            firing_rates, labels = resample(firing_rates, labels, replace=False, n_samples=len(labels_c2), random_state=42)

        # Standardize features
        scaler = StandardScaler()
        firing_rates_scaled = scaler.fit_transform(firing_rates.reshape(-1, 1))

        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_accuracies = []

        for train_index, test_index in skf.split(firing_rates_scaled, labels):
            X_train, X_test = firing_rates_scaled[train_index], firing_rates_scaled[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Fit Random Forest Classifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predict labels and calculate accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)

        # Determine classification label based on average accuracy across folds
        avg_accuracy = np.mean(fold_accuracies)
        if avg_accuracy > 0.6:  # Threshold for selectivity
            depth_classification[depth] = 'right-selective' if np.mean(firing_rates_c1) > np.mean(firing_rates_c2) else 'left-selective'
        else:
            depth_classification[depth] = 'neutral'

    return depth_classification