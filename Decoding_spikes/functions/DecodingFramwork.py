from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler



class DecodingFramework_OnCluster:
    def __init__(self, data_passive, data_active, labels_passive, labels_active, **kwargs):
        """
        Initialize the decoding framework.

        Parameters:
        -----------
        data_passive : ndarray
            Passive dataset of shape (n_trials, n_clusters, n_time_bins).
        data_active : ndarray
            Active dataset of shape (n_trials, n_clusters, n_time_bins).
        labels_passive : ndarray
            Labels for passive data.
        labels_active : ndarray
            Labels for active data.
        test_strategy : str
            Decoding procedure to use ('passive', 'active', 'both').
        n_components : int
            Number of PCA components to retain when feature_selection='pca'.
        feature_selection : str
            Feature selection method to use ('pca', 'average_clusters').
        n_time_bins : int, optional
            Number of time bins to include in the classification. If None, use all time bins.
        """
        # get parameters
 
        n_time_bins = kwargs.get('n_time_bins', None)
        self.data = data_passive [:, :, :n_time_bins] if n_time_bins else data_passive
        self.labels = labels_passive
        self.test_strategy = kwargs.get('test_strategy', 'both')
        self.n_components = kwargs.get('n_components', 5)
        self.feature_selection = kwargs.get('feature_selection', 'pca')
        self.external_data = data_active [:, :, :n_time_bins] if n_time_bins else data_active
        self.external_labels = labels_active
        self.n_folds =  kwargs.get('n_folds', 5)
        self.n_permutations = kwargs.get('n_permutations', 1000)

    def apply_feature_selection(self, X):
        """Apply feature selection based on the specified method."""
        if self.feature_selection == "pca":
            # Concatenate cluster and time bins
            n_trials, n_clusters, n_time_bins = X.shape
            X = X.reshape(n_trials, -1)  # Shape: (n_trials, n_clusters * n_time_bins)
            # Apply PCA
            pca = PCA(n_components=self.n_components)
            X_reduced = pca.fit_transform(X)
            return X_reduced, pca
        elif self.feature_selection == "average_clusters":
            # Average over clusters
            X_reduced = np.mean(X, axis=1)  # Shape: (n_trials, n_time_bins)
            return X_reduced, None
        else:
            raise ValueError(f"Unsupported feature selection method: {self.feature_selection}")

    def decode(self, **kwargs):
        """
        Perform decoding using logistic regression with specified test strategy and validate on permuted data.
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix for training.
        y : ndarray
            Labels for training data.
        n_folds : int
            Number of folds for cross-validation.
        n_permutations : int
            Number of permutations for generating the null distribution.
        
        Returns:
        --------
        results : dict
            Dictionary containing accuracy, null distribution, and p-value.
        """
        # get parameters
        n_permutations = kwargs.get('n_permutations', 1000)



        clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, class_weight='balanced')
        results = {}

        #############################
        ## step 1: prepare data 
        ##############################

        # Mask trials for right (1) vs no stimulus (0)
        mask_right_no_stim_passive = (self.labels == 1) | (self.labels == 0)
        X_right_no_stim_passive = self.data[mask_right_no_stim_passive, :, :]
        y_right_no_stim_passive = self.labels[mask_right_no_stim_passive]

        mask_right_no_stim_active = (self.external_labels == 1) | (self.external_labels == 0)
        X_right_no_stim_active = self.external_data[mask_right_no_stim_active, :, :]
        y_right_no_stim_active = self.external_labels[mask_right_no_stim_active]

        # Mask trials for left (-1) vs no stimulus (0)
        mask_left_no_stim_passive = (self.labels == -1) | (self.labels == 0)
        X_left_no_stim_passive = self.data[mask_left_no_stim_passive, :, :]
        y_left_no_stim_passive = self.labels[mask_left_no_stim_passive]

        mask_left_no_stim_active = (self.external_labels == -1) | (self.external_labels == 0)
        X_left_no_stim_active = self.external_data[mask_left_no_stim_active , :, :]
        y_left_no_stim_active = self.external_labels[mask_left_no_stim_active]

        #############################
        ## step 2: apply feature selection
        ##############################
        # Apply feature selection to the masked data
        X_reduced_right, _ = self.apply_feature_selection(X_right_no_stim_passive)
        X_reduced_left, _ = self.apply_feature_selection(X_left_no_stim_passive)
        external_data_reduced_right, _ = self.apply_feature_selection(X_right_no_stim_active)
        external_data_reduced_left, _ = self.apply_feature_selection(X_left_no_stim_active)


        # return X_reduced_right, X_reduced_left, y_right_no_stim_passive, y_left_no_stim_passive

        #############################
        ## step 3: compute true accuracy
        ##############################

        def compute_accuracy(X_train, y_train, X_test = None, y_test = None, skf=None):
            accuracies = []
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            if skf:   # apply cross-validation when the test strategy is passive, meaning test on part of trained data
                for train_idx, test_idx in skf.split(X_train, y_train):
                    clf.fit(X_train[train_idx], y_train[train_idx])
                    y_pred = clf.predict(X_train[test_idx])
                    accuracies.append(accuracy_score(y_train[test_idx], y_pred))
                return np.mean(accuracies)
            else: # do not apply cross-validation when the test strategy is active (or both), meaning test on external data
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                return accuracy_score(y_test, y_pred)


        # Determine training and testing sets based on test strategy
        if self.test_strategy == 'passive':  #  Train on passive data and test on passive data 
            skf = StratifiedKFold(n_splits= self.n_folds, shuffle=True, random_state=42)
            true_accuracy_right = compute_accuracy(X_train = X_reduced_right, y_train = y_right_no_stim_passive, skf = skf)
            true_accuracy_left = compute_accuracy(X_train =X_reduced_left,y_train = y_left_no_stim_passive, skf = skf)

        elif self.test_strategy == 'active': # Train on passive data and test on active data
            true_accuracy_right = compute_accuracy(X_train =X_reduced_right, y_train = y_right_no_stim_passive, X_test= external_data_reduced_right, y_test=  y_right_no_stim_active)
            true_accuracy_left = compute_accuracy(X_train = X_reduced_left, y_train = y_left_no_stim_passive, X_test = external_data_reduced_left, y_test=  y_left_no_stim_active)

        elif self.test_strategy == 'both': # train on passive data and test on passive data + active data
            # Split passive data into 80% train and 20% test randomly
            from sklearn.model_selection import train_test_split
            X_train_right, X_test_passive_right, y_train_right, y_test_passive_right = train_test_split(X_reduced_right, y_right_no_stim_passive, test_size=0.2, random_state=42, stratify=y_right_no_stim_passive)
            X_train_left, X_test_passive_left, y_train_left, y_test_passive_left = train_test_split(X_reduced_left, y_left_no_stim_passive, test_size=0.2, random_state=42, stratify=y_left_no_stim_passive)

            # Concatenate the rest of passive test data with active data
            X_combined_right = np.concatenate((X_test_passive_right, external_data_reduced_right), axis=0)
            y_combined_right = np.concatenate((y_test_passive_right, y_right_no_stim_active), axis=0)
            X_combined_left = np.concatenate((X_test_passive_left, external_data_reduced_left), axis=0)
            y_combined_left = np.concatenate((y_test_passive_left, y_left_no_stim_active), axis=0)

            # Train on 80% of passive data and test on combined rest + external data
            true_accuracy_right = compute_accuracy(X_train = X_train_right,  y_train =y_train_right,X_test=  X_combined_right, y_test= y_combined_right)
            true_accuracy_left = compute_accuracy(X_train = X_train_left,  y_train = y_train_left,X_test= X_combined_left, y_test= y_combined_left)

        results['true_accuracy_c1'] = true_accuracy_right
        results['true_accuracy_c2'] = true_accuracy_left

        #############################
        ## step 4: compute null distribution and p-value (validation)
        ##############################

        # Null distribution generation and p-value calculation
        def compute_null_distribution(X_train, y_train, X_test =None, y_test = None ,  skf=None):
            X_train = np.array(X_train)
            null_accuracies = []
            for _ in range(self.n_permutations):
                permuted_labels = np.random.permutation(y_train)
                if skf:     # passive (test)
                    permuted_accuracies = []
                    for train_idx, test_idx in skf.split(X_train, permuted_labels):
                        clf.fit(X_train[train_idx], permuted_labels[train_idx])
                        y_pred = clf.predict(X_train[test_idx])
                        permuted_accuracies.append(accuracy_score(permuted_labels[test_idx], y_pred))
                    null_accuracies.append(np.mean(permuted_accuracies))
                else: # active (test) or both (test)
                    
                    clf.fit(X_train, permuted_labels)
                    y_pred = clf.predict(X_test)
                    null_accuracies.append(accuracy_score(y_test, y_pred))
            return np.array(null_accuracies)
        
        if self.test_strategy == 'passive':
            null_accuracies_right = compute_null_distribution( X_reduced_right, y_right_no_stim_passive,  skf = skf)
            null_accuracies_left = compute_null_distribution( X_reduced_left, y_left_no_stim_passive , skf = skf)
        
        elif self.test_strategy == 'active':
            null_accuracies_right = compute_null_distribution( X_reduced_right, y_right_no_stim_passive, X_test= external_data_reduced_right, y_test= y_right_no_stim_active)
            null_accuracies_left = compute_null_distribution( X_reduced_left,y_left_no_stim_passive , X_test= external_data_reduced_left, y_test= y_left_no_stim_active)
        
        elif self.test_strategy == 'both':
            null_accuracies_right = compute_null_distribution( X_train_right, y_train_right, X_test= X_combined_right, y_test= y_combined_right)
            null_accuracies_left = compute_null_distribution( X_train_left, y_train_left, X_test= X_combined_left, y_test= y_combined_left)

        # Calculate p-values based on null distributions
        p_value_right = (np.sum(null_accuracies_right >= true_accuracy_right) + 1) / (len(null_accuracies_right) + 1)
        p_value_left = (np.sum(null_accuracies_left >= true_accuracy_left) + 1) / (len(null_accuracies_left) + 1)
        results['p_value_c1'] = p_value_right
        results['p_value_c2'] = p_value_left
        results['null_distribution_c1'] = null_accuracies_right
        results['null_distribution_c2'] = null_accuracies_left

        return results # include 'true_accuracy_right', 'true_accuracy_left', 'p_value_right', 'p_value_left', 'null_distribution_right', 'null_distribution_left'




class DecodingFramework_OnCluster_onlyActive:
    def __init__(self, data, labels, distanceTOchange = None,  **kwargs):

        # get parameters
 
        n_time_bins = kwargs.get('n_time_bins', None)
        self.data = data [:, :, :n_time_bins] if n_time_bins else data
        self.labels = labels
        self.test_strategy = kwargs.get('test_strategy', 'both')
        self.n_components = kwargs.get('n_components', 5)
        self.feature_selection = kwargs.get('feature_selection', 'pca')
        self.n_folds =  kwargs.get('n_folds', 5)
        self.n_permutations = kwargs.get('n_permutations', 1000)
        self.condition = kwargs.get('condition', 'right_left')
        self.distanceTOchange = distanceTOchange
        self.distance_threshold = kwargs.get('distance_threshold', 10)

    def apply_feature_selection(self, X):
        """Apply feature selection based on the specified method."""
        if self.feature_selection == "pca":
            # Concatenate cluster and time bins
            n_trials, n_clusters, n_time_bins = X.shape
            X = X.reshape(n_trials, -1)  # Shape: (n_trials, n_clusters * n_time_bins)
            # Apply PCA
            pca = PCA(n_components=self.n_components)
            X_reduced = pca.fit_transform(X)
            return X_reduced, pca
        elif self.feature_selection == "average_clusters":
            # Average over clusters
            X_reduced = np.mean(X, axis=1)  # Shape: (n_trials, n_time_bins)
            return X_reduced, None
        else:
            raise ValueError(f"Unsupported feature selection method: {self.feature_selection}")

    def decode(self, **kwargs):

        # get parameters
        n_permutations = kwargs.get('n_permutations', 1000)

        clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, class_weight='balanced')
        results = {}

        #############################
        ## step 1: prepare data 
        ##############################

        if self.condition == 'right_left':     # decode right vs left c1 = right, c2 = left
            # C1: first condition Mask trials for right (1) vs no stimulus (0) 
            mask_c1 = (self.labels == 1) | (self.labels == 0)
            # C2: second condition Mask trials for left (-1) vs no stimulus (0)
            mask_c2 = (self.labels == -1) | (self.labels == 0)

        elif self.condition == 'Blocks':       # decode blocks      0 = 0.5 prob left, 1 = 0.2 prob left, 2 = 0.8 prob left
            mask_c1 = [i for i,x in enumerate(self.labels) if x in [0, 1] and self.distanceTOchange[i] >self.distance_threshold] # block right vs no blocks
            mask_c2 = [i for i,x in enumerate(self.labels) if x in [0, 2] and self.distanceTOchange[i] >self.distance_threshold] # block left vs no blocks


        X_c1 = self.data[mask_c1, :, :]
        y_c1 = self.labels[mask_c1]
        X_c2 = self.data[mask_c2, :, :]
        y_c2 = self.labels[mask_c2]

  
        #############################
        ## step 2: apply feature selection
        ##############################
        # Apply feature selection to the masked data
        X_reduced_c1, _ = self.apply_feature_selection(X_c1)
        X_reduced_c2, _ = self.apply_feature_selection(X_c2)

        #############################
        ## step 3: compute true accuracy
        ##############################

        def compute_accuracy(X_train, y_train,  skf):
            accuracies = []
            X_train = np.array(X_train)
            y_train = np.array(y_train)
     
            for train_idx, test_idx in skf.split(X_train, y_train):

                X_train_fold, X_test_fold = X_train[train_idx], X_train[test_idx]
                y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]

                # Apply scaling
                scaler = StandardScaler()
                X_train_fold = scaler.fit_transform(X_train_fold)
                X_test_fold = scaler.transform(X_test_fold)


                # Fit the model and predict
                clf.fit(X_train_fold, y_train_fold)
                y_pred = clf.predict(X_test_fold)
                
                # Compute accuracy
                accuracies.append(accuracy_score(y_test_fold, y_pred))

            return np.mean(accuracies)

        skf = StratifiedKFold(n_splits= self.n_folds, shuffle=True, random_state=42)
        true_accuracy_c1 = compute_accuracy(X_train = X_reduced_c1, y_train = y_c1, skf = skf)
        true_accuracy_c2 = compute_accuracy(X_train =X_reduced_c2,y_train = y_c2, skf = skf)

        results['true_accuracy_c1'] = true_accuracy_c1
        results['true_accuracy_c2'] = true_accuracy_c2

        #############################
        ## step 4: compute null distribution and p-value (validation)
        ##############################
        
        def compute_null_distribution(X_train, y_train, skf):
            X_train = np.array(X_train)
            null_accuracies = []
            
            for _ in range(n_permutations):
                # Permute the labels
                permuted_labels = np.random.permutation(y_train)
                permuted_accuracies = []
                
                for train_idx, test_idx in skf.split(X_train, permuted_labels):
                    # Split the data
                    X_train_fold, X_test_fold = X_train[train_idx], X_train[test_idx]
                    permuted_train_labels = permuted_labels[train_idx]
                    permuted_test_labels = permuted_labels[test_idx]
                    
                    # Apply scaling
                    scaler = StandardScaler()
                    X_train_fold = scaler.fit_transform(X_train_fold)
                    X_test_fold = scaler.transform(X_test_fold)
                    
                    # Fit the model and predict
                    clf.fit(X_train_fold, permuted_train_labels)
                    y_pred = clf.predict(X_test_fold)
                    
                    # Compute accuracy
                    permuted_accuracies.append(accuracy_score(permuted_test_labels, y_pred))
                
                # Store the mean accuracy for this permutation
                null_accuracies.append(np.mean(permuted_accuracies))
            
            return np.array(null_accuracies)

        null_accuracies_c1 = compute_null_distribution( X_reduced_c1, y_c1,  skf = skf)
        null_accuracies_c2 = compute_null_distribution( X_reduced_c2, y_c2 , skf = skf)

        # Calculate p-values based on null distributions
        p_value_c1 = (np.sum(null_accuracies_c1 >= true_accuracy_c1) + 1) / (len(null_accuracies_c1) + 1)
        p_value_c2 = (np.sum(null_accuracies_c2 >= true_accuracy_c2) + 1) / (len(null_accuracies_c2) + 1)
        results['p_value_c1'] = p_value_c1
        results['p_value_c2'] = p_value_c2
        results['null_distribution_c1'] = null_accuracies_c1
        results['null_distribution_c2'] = null_accuracies_c2

        return results # include 'true_accuracy_right', 'true_accuracy_left', 'p_value_right', 'p_value_left', 'null_distribution_right', 'null_distribution_left'

