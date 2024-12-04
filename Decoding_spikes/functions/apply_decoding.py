from functions import DecodingFramework_OnCluster, pre_processed_active_data, pre_processed_passive_data
import numpy as np
import pickle
import pandas as pd
import datetime
def Apply_decoding(eid, pid, PARAMS_preprocess, PARAMS_decoding, save = True, save_path = None):

    ###########################
    # Preprocess data
    ###########################
    pre_processed_data_active = pre_processed_active_data(eid, pid, **PARAMS_preprocess)
    pre_processed_data_passive = pre_processed_passive_data(eid, pid, **PARAMS_preprocess)

    FR_channels_active = pre_processed_data_active['firing_rates']
    FR_channels_passive = pre_processed_data_passive['firing_rates']

    trial_info_active = pre_processed_data_active['trial_info']
    trial_info_passive = pre_processed_data_passive['trial_info']

    channel_info = pre_processed_data_active['channel_info']


    ###########################
    # Decoding per channel 
    ###########################
    All_results = {}
    for i, channel in enumerate(channel_info['ch_indexs'].values):
        try:
            print(f'processing channel {i}/{len(channel_info["ch_indexs"].values)}')
            data_active = FR_channels_active[channel] # (n_trials, n_clusters, n_time_bins)
            data_passive = FR_channels_passive[channel]
            labels_active = trial_info_active['labels']
            labels_passive = trial_info_passive['labels']
            decoder = DecodingFramework_OnCluster(data_passive, data_active, labels_passive, labels_active, **PARAMS_decoding) 
            All_results[channel] = decoder.decode() # include 'true_accuracy_right', 'true_accuracy_left', 'p_value_right', 'p_value_left', 'null_distribution_right', 'null_distribution_left'
        except Exception as e:
            print(f'error in channel {i}')
            print(e)
            return
    decoding_results = pd.DataFrame(All_results)
    decoding_results = decoding_results.T
    decoding_results.reset_index(level=0, inplace=True)
    # add metadata to the results
    all_data = {'channel_info': channel_info, 'PARAMS_preprocess': PARAMS_preprocess, 'PARAMS_decoding': PARAMS_decoding, 'eid': eid, 'pid': pid, 'decoding_results': decoding_results}
    
    ##############
    # Save results
    ##############
    if save:
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(all_data, f)
            return save_path
        else:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(f'results/{pid}_{current_time}.pkl', 'wb') as f:
                pickle.dump(decoding_results, f)
            return f'results/{pid}_{current_time}.pkl'
    else:
        return all_data