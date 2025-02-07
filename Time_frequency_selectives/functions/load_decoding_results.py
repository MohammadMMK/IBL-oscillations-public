

import os 
import pickle
import pandas as pd

def load_decoding_results(pid_eid_pairs, suffix , dir = '/mnt/data/AdaptiveControl/mohammad/crnl/ibl_oscillations/_analyses/Decoding_spikes/results/rightVsLeft' ):
    """
    Load decoding results from specified directory and return a flattened DataFrame.
    Parameters:
    pid_eid_pairs (list of tuples): List of (pid, eid) pairs to load decoding results for.
    suffix (str): Suffix to append to the pid to form the filename.
    dir (str, optional): Directory where the decoding results are stored. Default is '/mnt/data/AdaptiveControl/mohammad/crnl/ibl_oscillations/_analyses/Decoding_spikes/results/rightVsLeft'.
    Returns:
    pd.DataFrame: Flattened DataFrame containing the decoding results with the following columns:
        - accuracies_c1: List of accuracies for condition 1.
        - accuracies_c2: List of accuracies for condition 2.
        - p_value_c1: List of p-values for condition 1.
        - p_value_c2: List of p-values for condition 2.
        - pid: List of pids repeated for each channel.
        - ch_indexs: List of channel indices.
        - acronyms: List of acronyms for the channels.
    Prints:
    Number of pids without decoding result.
    Number of total channels in the flattened results.
    """
    
    results = {
            "accuracies_c1": [],
            "accuracies_c2": [],
            "p_value_c1": [],
            "p_value_c2": [],
            "pid": [],
            "ch_indexs": [],
            "acronyms": []
        }
    j = 0
    for pid, eid in pid_eid_pairs:
        save_path = f"{dir}/{pid}_{suffix}.pkl"  #     all_data = {'channel_info': channel_info, 'PARAMS_preprocess': PARAMS_preprocess, 'PARAMS_decoding': PARAMS_decoding, 'eid': eid, 'pid': pid, 'decoding_results': decoding_results}
        if os.path.exists(save_path):
                with open(save_path, 'rb') as f:
                    all_data = pickle.load(f)
                decoding_results = all_data['decoding_results']
                results["accuracies_c1"].append(decoding_results['true_accuracy_c1'].values)
                results["accuracies_c2"].append(decoding_results['true_accuracy_c2'].values)
                results["p_value_c1"].append(decoding_results['p_value_c1'].values)
                results["p_value_c2"].append(decoding_results['p_value_c2'].values)
                # repeat pid
                channel_info = all_data['channel_info'] 
                pids = [pid] * len(channel_info['ch_indexs'])
                results["pid"].append(pids)
                results["ch_indexs"].append(channel_info['ch_indexs'])
                results["acronyms"].append(channel_info['acronyms'])
        else:
            j+=1
    print(f"Number of pids without decoding result: {j}")
    results = pd.DataFrame(results)

    # flatten results
    flat_results = {}
    for column in results.columns:
        list_column = results[column].values
        flat_column = [item for sublist in list_column for item in sublist]
        flat_results[column] = flat_column
    flat_results = pd.DataFrame(flat_results)
    print(f' number of total channnels {len(flat_results)}')

    return flat_results

def selective_channels(flat_results, p_value_threshold = 0.05, accuracy_threshold = 0.6):
    """
    Identifies selective and sensitive channels based on p-value and accuracy thresholds.
    Parameters:
    flat_results (DataFrame): A pandas DataFrame containing the results with columns 'p_value_c1', 'p_value_c2', 'accuracies_c1', and 'accuracies_c2'.
    p_value_threshold (float, optional): The threshold for p-values to determine significance. Default is 0.05.
    accuracy_threshold (float, optional): The threshold for accuracies to determine sensitivity. Default is 0.6.
    Returns:
    tuple: A tuple containing five DataFrames:
        - right_selective: Channels that are right selective.
        - left_selective: Channels that are left selective.
        - right_sensitive: Channels that are right sensitive.
        - left_sensetive: Channels that are left sensitive.
        - neutral: Channels that are neutral.
    """
  
    right_sensitive = flat_results[(flat_results['p_value_c1'] < p_value_threshold) & (flat_results['accuracies_c1'] > accuracy_threshold)]
    right_selective = right_sensitive[(right_sensitive['p_value_c2'] > p_value_threshold) | (right_sensitive['accuracies_c2'] < accuracy_threshold)]
    left_sensetive = flat_results[(flat_results['p_value_c2'] < p_value_threshold) & (flat_results['accuracies_c2'] > accuracy_threshold)]
    left_selective = left_sensetive[(left_sensetive['p_value_c1'] > p_value_threshold) | (left_sensetive['accuracies_c1'] < accuracy_threshold)]
    neutral = flat_results[(flat_results['p_value_c1'] > p_value_threshold) | (flat_results['p_value_c2'] > p_value_threshold) | (flat_results['accuracies_c1'] < accuracy_threshold) | (flat_results['accuracies_c2'] < accuracy_threshold)]
    return right_selective, left_selective, right_sensitive, left_sensetive , neutral