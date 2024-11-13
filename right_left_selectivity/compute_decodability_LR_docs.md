#  Decodability Computation with Logistic Regression


## Function Overview

### `compute_decodability_LR`

The `compute_decodability_LR` function applies Logistic Regression to measure the decodability between two conditions. It uses K-Fold cross-validation to ensure a robust evaluation of the classification performance at each depth and time point.

- **Parameters**:
  - `z_score_condition1` (numpy.ndarray): A 3D array of z-scored firing rates for condition 1 with shape `(n_c1_trials, n_depths, n_time)`.
  - `z_score_condition2` (numpy.ndarray): A 3D array of z-scored firing rates for condition 2 with shape `(n_c2_trials, n_depths, n_time)`.
  - `k_folds` (int): Number of folds for K-Fold cross-validation. Default is `5`.

- **Returns**:
  - `decodability_scores` (numpy.ndarray): A 2D array of decodability scores (Area Under ROC Curve - AUROC) with shape `(n_depths, n_time)`. Each entry represents the decodability score at a specific depth and time point, ranging from 0.5 (chance level) to 1 (perfect decodability).

## Methodology

The function performs the following steps to compute decodability:

1. **Data Preparation**:
   - Labels are assigned to each trial for the two conditions, with `1` for trials in `condition1` and `0` for trials in `condition2`.
   - The z-scored firing rates for each condition are concatenated to form a combined dataset with associated labels.

2. **Class Imbalance Handling**:
   - In cases where there is a class imbalance between the two conditions, resampling is used to balance the number of trials from each condition.

3. **Feature Scaling**:
   - Each depth and time point's firing rates are standardized to have a mean of zero and a standard deviation of one using `StandardScaler`.

4. **Cross-Validation and Model Fitting**:
   - K-Fold cross-validation is performed (default of 5 folds).
   - For each fold, Logistic Regression is used to classify the conditions.
   - The probability scores for the test data are evaluated using the Area Under the Receiver Operating Characteristic (AUROC) curve, indicating how well the model distinguishes between the two conditions.

5. **AUROC Averaging**:
   - For each depth and time point, the AUROC scores across folds are averaged to produce a robust decodability score for that specific depth-time combination.

## Example Usage

```python
from decodability import compute_decodability_LR
import numpy as np

# Example data: Generate dummy z-scored firing rate data
z_score_condition1 = np.random.randn(100, 10, 20)  # 100 trials, 10 depths, 20 time points
z_score_condition2 = np.random.randn(80, 10, 20)   # 80 trials, 10 depths, 20 time points

# Compute decodability
decodability_scores = compute_decodability_LR(z_score_condition1, z_score_condition2, k_folds=5)

# `decodability_scores` will have shape (10, 20), representing decodability for each depth and time point.
```
## Dependencies
numpy: For handling arrays and numerical data.
sklearn.model_selection.StratifiedKFold: For stratified K-Fold cross-validation.
sklearn.linear_model.LogisticRegression: For performing logistic regression.
sklearn.preprocessing.StandardScaler: For standardizing feature data.
sklearn.metrics.roc_auc_score: For computing the Area Under ROC Curve (AUROC).

## Notes
Interpretation of Decodability Scores: A decodability score close to 0.5 suggests that the model cannot distinguish between the two conditions (chance level), while a score closer to 1.0 indicates high decodability.
Cross-Validation: K-Fold cross-validation helps prevent overfitting and provides a reliable estimate of model performance.
Class Imbalance: The function includes resampling for cases where there is an imbalance between the number of trials for each condition, enhancing the robustness of the result
