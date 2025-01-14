

import os 
import pickle
import pandas as pd

def load_decoding_results(pid_eid_pairs, suffix , dir = '/mnt/data/AdaptiveControl/mohammad/crnl/ibl_oscillations/_analyses/Decoding_spikes/results/rightVsLeft' ):
    
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
    print(f"Number of missing pids: {j}")
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
    c1_sensitive = flat_results[(flat_results['p_value_c1'] < p_value_threshold) & (flat_results['accuracies_c1'] > accuracy_threshold)]
    c1_selective = c1_sensitive[(c1_sensitive['p_value_c2'] > p_value_threshold) | (c1_sensitive['accuracies_c2'] < accuracy_threshold)]
    c2_sensetive = flat_results[(flat_results['p_value_c2'] < p_value_threshold) & (flat_results['accuracies_c2'] > accuracy_threshold)]
    c2_selective = c2_sensetive[(c2_sensetive['p_value_c1'] > p_value_threshold) | (c2_sensetive['accuracies_c1'] < accuracy_threshold)]
    neutral = flat_results[(flat_results['p_value_c1'] > p_value_threshold) | (flat_results['p_value_c2'] > p_value_threshold) | (flat_results['accuracies_c1'] < accuracy_threshold) | (flat_results['accuracies_c2'] < accuracy_threshold)]
    return c1_selective, c2_selective, c1_sensitive, c2_sensetive , neutral