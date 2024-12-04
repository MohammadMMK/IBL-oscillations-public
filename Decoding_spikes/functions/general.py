
def monitor_job_status(all_jobs):
    # Monitor job status
    total_jobs = len(all_jobs)
    print(f"Submitted {total_jobs} jobs.")
    finished_jobs = 0
    completed_jobs_set = set()
    while finished_jobs < total_jobs:
        for idx, job in enumerate(all_jobs):
            if job.done() and idx not in completed_jobs_set:
                finished_jobs += 1
                completed_jobs_set.add(idx)
                print(f"Jobs finished: {finished_jobs}/{total_jobs}")
    print("All jobs are finished.")

import numpy as np
import pandas as pd
import submitit
import pickle

def load_decoded_results(parameters, all_jobs=None, pickle_paths=None):
    """
    Load decoded results from distributed job objects or pickle files.
    """
    results = {
        "accuracies_right": [],
        "accuracies_left": [],
        "pvalues_right": [],
        "pvalues_left": []
    }

    # Helper function to process decoding results
    def process_data(all_data, idx):
        try:
            decoding_results = all_data['decoding_results']
            results["accuracies_right"].append(decoding_results['true_accuracy_right'].values)
            results["accuracies_left"].append(decoding_results['true_accuracy_left'].values)
            results["pvalues_right"].append(decoding_results['p_value_right'].values)
            results["pvalues_left"].append(decoding_results['p_value_left'].values)
        except Exception as e:
            print(f"Error processing data at index {idx}")
            nonlocal parameters
            parameters = np.delete(parameters, np.where(parameters == idx))

    # Process jobs
    if all_jobs:
        for i, job in enumerate(all_jobs):
            try:
                all_data = job.result()
                if all_data is None:
                    raise ValueError(f"Job {i} returned None.")
                process_data(all_data, i)
            except Exception as e:
                print(f"Job {i} failed: {e}")

    # Process pickle files
    if pickle_paths:
        for i, path in enumerate(pickle_paths):
            try:
                with open(path, 'rb') as f:
                    all_data = pickle.load(f)
                process_data(all_data, i)
            except Exception as e:
                print(f"Failed to load {path}: {e}")

    # Convert results to arrays
    results = {key: np.array(value) for key, value in results.items()}
    channel_info = all_data.get('channel_info', None) if 'all_data' in locals() else None

    return (results["accuracies_right"], results["accuracies_left"],
            results["pvalues_right"], results["pvalues_left"],
            parameters, channel_info)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

def plot_decoder_accuracies(all_accuracies_right, all_accuracies_left, 
                            parameters, channel_info, 
                            all_pvalues_right, all_pvalues_left,
                            accuracy_threshold=None, pvalue_threshold=None,
                            dark_background=None, line_width=2.0, title='title', short_xticks= False):
    """
    Plot decoder accuracies for right and left stimuli across parameters.

    Args:
        all_accuracies_right (ndarray): Decoder accuracies for right stimuli, shape (n_parameters, n_channels).
        all_accuracies_left (ndarray): Decoder accuracies for left stimuli, shape (n_parameters, n_channels).
        parameters (list or ndarray): List of parameter values corresponding to the accuracy data.
        channel_info (dict): Dictionary containing channel information, including 'acronym'.
        all_pvalues_right (ndarray): P-values for right stimuli, shape (n_parameters, n_channels).
        all_pvalues_left (ndarray): P-values for left stimuli, shape (n_parameters, n_channels).
        accuracy_threshold (float): Minimum accuracy to include channels in the plot.
        pvalue_threshold (float): Maximum p-value to include channels in the plot.
        dark_background (bool): Whether to use a dark background for the plots.
        line_width (float): Width of the lines in the parallel coordinates plots.
        title (str): Title for the entire figure.
    """
    # Ensure parameters is a list of strings
    parameters = [str(param) for param in parameters]

    # Extract channel names
    channel_names = channel_info['acronyms']

    # Apply separate accuracy and p-value thresholds for right and left channels
    selected_channels_right = np.ones(all_accuracies_right.shape[1], dtype=bool)
    selected_channels_left = np.ones(all_accuracies_left.shape[1], dtype=bool)

    if accuracy_threshold is not None:
        selected_channels_right &= all_accuracies_right.max(axis=0) >= accuracy_threshold
        selected_channels_left &= all_accuracies_left.max(axis=0) >= accuracy_threshold

    if pvalue_threshold is not None:
        selected_channels_right &= all_pvalues_right.min(axis=0) <= pvalue_threshold
        selected_channels_left &= all_pvalues_left.min(axis=0) <= pvalue_threshold

    # Filter data and channel names
    filtered_accuracies_right = all_accuracies_right[:, selected_channels_right]
    filtered_accuracies_left = all_accuracies_left[:, selected_channels_left]

    filtered_channel_names_right = [channel_names[i] for i in np.where(selected_channels_right)[0]]
    filtered_channel_names_left = [channel_names[i] for i in np.where(selected_channels_left)[0]]

    # Ensure parameters align with the filtered data
    filtered_parameters = parameters[:filtered_accuracies_right.shape[0]]

    # Prepare data for plotting
    data_right = pd.DataFrame(filtered_accuracies_right.T, columns=filtered_parameters)
    data_right['Channel'] = filtered_channel_names_right
    data_left = pd.DataFrame(filtered_accuracies_left.T, columns=filtered_parameters)
    data_left['Channel'] = filtered_channel_names_left

    # Set dark background if requested
    with plt.style.context('dark_background') if dark_background else plt.style.context('default'):
        # Create plot
        plt.figure(figsize=(15, 6))

        # Plot for left stimulus (First column)
        plt.subplot(1, 2, 1)
        parallel_coordinates(data_left, class_column='Channel', colormap='plasma', alpha=0.7, linewidth=line_width)
        plt.xlabel('Parameters')
        plt.ylabel('Accuracy (%)')
        if short_xticks:
            plt.xticks(ticks=np.arange(0, len(filtered_parameters), 5), labels=filtered_parameters[::5])
        plt.title('Left Stimulus Across Parameters')
        plt.grid(True)

        # Plot for right stimulus (Second column)
        plt.subplot(1, 2, 2)
        parallel_coordinates(data_right, class_column='Channel', colormap='viridis', alpha=0.7, linewidth=line_width)
        plt.xlabel('Parameters')
        plt.ylabel('Accuracy (%)')
        if short_xticks:
            plt.xticks(ticks=np.arange(0, len(filtered_parameters), 5), labels=filtered_parameters[::5])
        plt.title('Right Stimulus Across Parameters')
        plt.grid(True)

        # Add a global title
        plt.suptitle(f"{title}\nAccuracy threshold: {accuracy_threshold}, P-value threshold: {pvalue_threshold}", 
                     fontsize=16, y=1.02)

        plt.tight_layout()
        plt.show()
