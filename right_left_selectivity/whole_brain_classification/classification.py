import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

def compute_channel_decodability(df, n_splits=5, random_state=42):
    """
    Compute the decodability of right vs left stimuli from firing rates for each channel,
    handling imbalanced classes with class weights and using AUROC as the scoring metric.
    
    Parameters:
    - df: DataFrame with firing rate and metadata for each channel
    - n_splits: int, number of splits for cross-validation
    - random_state: int, seed for reproducibility
    
    Returns:
    - decodability_scores: DataFrame with channel-wise decodability results, including x, y, z, and acronyms
    """
    
    # Prepare an empty list to collect decodability results
    results = []

    # Get unique channels from the data
    channels = df['ch_index'].unique()

    for ch in channels:
        # Filter data for the current channel
        channel_data = df[df['ch_index'] == ch]

        # Extract firing rates, labels, and metadata
        X = channel_data['firing_rate'].values.reshape(-1, 1)  # Reshape for scikit-learn
        y = channel_data['label'].values
        x_coord = channel_data['x'].iloc[0]
        y_coord = channel_data['y'].iloc[0]
        z_coord = channel_data['z'].iloc[0]
        acronym = channel_data['acronym'].iloc[0]

        # Calculate class weights based on label distribution
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weights_dict = {i: weight for i, weight in zip(np.unique(y), class_weights)}

        # Initialize cross-validation and logistic regression model with class weights
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        model = LogisticRegression(class_weight=weights_dict)

        # Perform cross-validation to assess decodability using AUROC
        scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

        # Collect mean AUROC score and standard deviation
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Append results for the current channel, including coordinates and acronyms
        results.append({
            'ch_index': ch,
            'x': x_coord,
            'y': y_coord,
            'z': z_coord,
            'acronym': acronym,
            'decodability_auroc_mean': mean_score,
            'decodability_auroc_std': std_score
        })

    # Convert results to DataFrame for easier analysis
    decodability_scores = pd.DataFrame(results)
    return decodability_scores