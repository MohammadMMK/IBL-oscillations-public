from functions import DecodingFramework_OnCluster, pre_processed_active_data, pre_processed_passive_data
pid = 'a8a59fc3-a658-4db4-b5e8-09f1e4df03fd'
eid = '5ae68c54-2897-4d3a-8120-426150704385'
pre_processed_data_active = pre_processed_active_data(eid, pid, **PARAMS_preprocess)
pre_processed_data_passive = pre_processed_passive_data(eid, pid, **PARAMS_preprocess)

FR_channels_active = pre_processed_data_active['firing_rates']
FR_channels_passive = pre_processed_data_passive['firing_rates']

trial_info_active = pre_processed_data_active['trial_info']
trial_info_passive = pre_processed_data_passive['trial_info']

channel_info = pre_processed_data_active['channel_info']
print(f"number of channels {len(channel_info['ch_indexs'])}")
print(f"number of active trials {len(trial_info_active['labels'])}")
print(f"number of passive trials {len(trial_info_passive['labels'])}")
print('trials in passive')
print('right')
print(len([i for i in trial_info_passive['labels'].values if i == 1]))
print('left')
print(len([i for i in trial_info_passive['labels'].values if i == -1]))
print('nostim')
print(len([i for i in trial_info_passive['labels'].values if i == 0]))

All_results = {}
for i, channel in enumerate(channel_info['ch_indexs'].values):
    print(f'processing channel {i}/{len(channel_info["ch_indexs"].values)}')
    data_active = FR_channels_active[channel] # (n_trials, n_clusters, n_time_bins)
    data_passive = FR_channels_passive[channel]
    labels_active = trial_info_active['labels']
    labels_passive = trial_info_passive['labels']
    print(f'number of active trials {len(labels_active)}')
    print(f'number of passive trials {len(labels_passive)}')
    print(f'shape of data active {data_active.shape}')
    print(f'shape of data passive {data_passive.shape}')
    decoder = DecodingFramework_OnCluster(data_passive, data_active, labels_passive, labels_active, **PARAMS_decoding) 
    All_results[channel] = decoder.decode() # include 'true_accuracy_right', 'true_accuracy_left', 'p_value_right', 'p_value_left', 'null_distribution_right', 'null_distribution_left'
    break
# decoder = DecodingFramework_OnCluster(data_passive, data_active, labels_passive, labels_active, **PARAMS_decoding) 
# All_results[channel] = decoder.decode() # include 'true_accuracy_right', 'true_accuracy_left', 'p_value_right', 'p_value_left', 'null_distribution_right', 'null_distribution_left'
