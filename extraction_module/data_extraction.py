

def data_extraction( eid, probe_label, **kwargs):

    print('eid:', eid)
    print('probe_label:', probe_label)
    print('kwargs:', kwargs)

    from one.api import ONE
    ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
    one = ONE(password='international')
    from detect_bad_channels import detect_bad_channels
    import os 
    from config import  BRAIN_REGION_PAIRS, SUBMITIT_PARAMS, extraction_parameters
    import pandas as pd
    import numpy as np
    import mne
    from brainbox.io.one import SpikeSortingLoader
    from iblatlas.atlas import AllenAtlas
    from iblatlas.regions import BrainRegions
    import ibldsp.voltage as voltage
    from pprint import pprint
    import spikeglx



    extract_wheel = kwargs.get('extract_wheel', False)
    extract_dlc = kwargs.get('extract_dlc', False)
    extract_spikes = kwargs.get('extract_spikes', False)
    extract_lfp = kwargs.get('extract_lfp', False)
    overwrite_behavior = kwargs.get('overwrite_behavior', False)
    overwrite_wheel = kwargs.get('overwrite_wheel', False)
    overwrite_dlc = kwargs.get('overwrite_dlc', False)
    overwrite_spikes = kwargs.get('overwrite_spikes', False)
    overwrite_lfp = kwargs.get('overwrite_lfp', False)
    resampled_fs = kwargs.get('resampled_fs', 500)
       
       
    ##############################
    # Define paths to save/load data
    ##############################
    pids, labels = one.eid2pid(eid)
    pid = next((pid for pid, label in zip(pids, labels) if label == probe_label), None)
    print(f'Extracting data for eid {eid} and probe {pid}...')
    non_ephys_path = f'/mnt/data/AdaptiveControl/IBLrawdata/eid_data/{eid}'
    ephys_path = f'/mnt/data/AdaptiveControl/IBLrawdata/pid_data/{pid}'

    if not os.path.isdir(non_ephys_path):
        os.makedirs(non_ephys_path)
    if not os.path.isdir(ephys_path):
        os.makedirs(ephys_path)  

    # File paths for behavior, wheel, DLC, spikes, and LFP data
    file_path_be = os.path.join(non_ephys_path, f'trials_{eid}.pkl')
    file_path_wh = os.path.join(non_ephys_path, f'wheel_{eid}.pkl')
    file_path_dlc = os.path.join(non_ephys_path, f'dlc_{eid}.pkl')
    file_path_licks = os.path.join(non_ephys_path, f'licks_{eid}.pkl')
    file_path_spikes = os.path.join(ephys_path, f'spikes_{pid}.pkl')
    file_path_lfp = os.path.join(ephys_path, f'lfp_{pid}_raw.fif')
    file_path_probe = os.path.join(non_ephys_path,  f'probe_{pid}.pkl')

    ##############################
    # Extract behavior data
    ##############################
    print('Extracting behavior data...')
    
    if not os.path.isfile(file_path_be) or overwrite_behavior:
        behavior = one.load_object(eid, 'trials', collection='alf')
        # behavior = one.load_dataset(eid, 'alf/_ibl_trials.table.pqt')
        behavior_df = behavior.to_df()
        behavior_df.to_pickle(file_path_be)
    else:
        behavior_df = pd.read_pickle(file_path_be)
        print(f'Trials data already exists for {eid}')


    ##############################
    # Extract wheel data
    ##############################
    if extract_wheel:
        print('Extracting wheel data...')
        try:
            if not os.path.isfile(file_path_wh) or overwrite_wheel:
                wheel = one.load_object(eid, 'wheel', collection='alf')
                wheel_df = wheel.to_df()
                wheel_df.to_pickle(file_path_wh)
                del wheel
                del wheel_df
            else:
                print(f'Wheel data already exists for {eid}')
        except Exception as e:
            print(f"Error extracting wheel data for eid {eid}")

    ##############################
    # Extract DLC and lick data
    ##############################
    if extract_dlc:
        print('Extracting DLC and lick data...')
        try:
            if not os.path.isfile(file_path_dlc) or overwrite_dlc:
                # Load DLC data
                leftDLC = one.load_object(eid, 'leftCamera', collection='alf')
                rightDLC = one.load_object(eid, 'rightCamera', collection='alf')
                DLC_df = pd.DataFrame({
                    'times': [leftDLC['times']],
                    'nose_tip': [leftDLC['dlc'][['nose_tip_x', 'nose_tip_y', 'nose_tip_likelihood']]],
                    'paw_l': [leftDLC['dlc'][['paw_l_x', 'paw_l_y', 'paw_l_likelihood']]],
                    'paw_r': [rightDLC['dlc'][['paw_r_x', 'paw_r_y', 'paw_r_likelihood']]],
                    'pupildiameter_l': [leftDLC['features']['pupilDiameter_smooth']],
                    'pupildiameter_r': [rightDLC['features']['pupilDiameter_smooth']]
                })
                DLC_df.to_pickle(file_path_dlc)

                # Load lick data
                licks = one.load_object(eid, 'licks', collection='alf')
                licks_df = pd.DataFrame(licks)
                licks_df.to_pickle(file_path_licks)

                # Cleanup
                del leftDLC, rightDLC, DLC_df, licks, licks_df
            else:
                print(f'DLC data already exists for {eid}')
        except Exception as e:
            print(f"Error extracting DLC data for eid {eid}")

    ##############################
    # Extract ephys data (Spikes)
    ##############################
    if extract_spikes:

        print('Extracting ephys data (spikes)...')
        try:
            ssl = SpikeSortingLoader(pid=pid, one=one)
        except Exception as e:
            print("Error initializing SpikeSortingLoader")
            return False

        try:
            if not os.path.isfile(file_path_spikes) or overwrite_spikes:
                spike_spikes, spike_clusters, spike_channels = ssl.load_spike_sorting()
                spikes_channels_indices = spike_clusters['channels'][spike_spikes['clusters']]
                spikes_df = pd.DataFrame({'spikes': spike_spikes, 'clusters': spike_clusters})
                spikes_df.to_pickle(file_path_spikes)

                # Cleanup
                del spikes_df, spike_spikes, spike_channels, spike_clusters, spikes_channels_indices
            else:
                print(f'Spikes data already exists for {pid}')
        except Exception as e:
            print(f"Error extracting spike data for pid {pid}")

    ##############################
    # Extract ephys data (LFP) 
    ##############################
    if extract_lfp:
        print('Extracting LFP data...')
        if not os.path.isfile(file_path_lfp) or overwrite_lfp:
            lfpdata = None
            ssl = SpikeSortingLoader(pid=pid, one=one)
            # Load LFP raw data
            dsets = one.list_datasets(eid, collection=f'raw_ephys_data/{probe_label}', filename='*.lf.*')
            data_files, info = one.load_datasets(eid, dsets, download_only=False)
            pprint(info[0])
            bin_file = next(df for df in data_files if df.suffix == '.cbin')
            sr_lf = spikeglx.Reader(bin_file)

            # Define intervals based on behavior data
            intervals = np.array([behavior_df['intervals_0'], behavior_df['intervals_1']]).T
            start_time = intervals[0][0] - 1
            last_time = intervals[-1][1] + 1
            first, last = ssl.samples2times([start_time, last_time], direction='reverse') / 12
            print(f'Extracting LFP data from {first} to {last} seconds...')

            # Process LFP in chunks
            lfp_nchunks = 25
            sliced_intervals = np.linspace(first, last, lfp_nchunks)
            for i in range(len(sliced_intervals) - 1):
                print(f'Processing chunk {i+1}/{lfp_nchunks}...')
                tsel = slice(int(sliced_intervals[i]), int(sliced_intervals[i+1]))
                raw = sr_lf[tsel, :-sr_lf.nsync].T
                destriped = voltage.destripe_lfp(raw.astype(float), fs=sr_lf.fs)
                decimated = mne.filter.resample(destriped.astype(float), up=2.0, down=10.0, window='boxcar', npad='auto', pad='reflect_limited', verbose=False)

                if lfpdata is None:
                    lfpdata = decimated
                else:
                    lfpdata = np.concatenate((lfpdata, decimated), axis=1)

                # Cleanup
                del destriped, decimated

            # Save LFP data in MNE format
            
            probe_df = pd.DataFrame(ssl.load_channels())
            channel_names_ordered = [f"{probe_df['acronym'][i]}_{i}" for i in range(len(probe_df))]
            mne_info = mne.create_info(channel_names_ordered, sr_lf.fs/5, ch_types='seeg')
            lfp_mne = mne.io.RawArray(lfpdata, mne_info, first_samp=start_time*(sr_lf.fs/5)) # we will use the first sample as the start time to aligne the events when we add annotations

            # Detect bad channels
            data2label = lfp_mne.get_data(start=10000, stop=min(200000, lfp_mne.last_samp))
            channel_labels, channel_features = detect_bad_channels(data2label, resampled_fs)
            bad_channel_index = np.where(np.logical_or(channel_labels == 1, channel_labels == 2))[0]
            bad_channel_name = [lfp_mne.info['ch_names'][i] for i in bad_channel_index]
            lfp_mne.info['bads'] = bad_channel_name


            # Save LFP data and probe information
            print('Saving LFP data...') 
            lfp_mne.save(file_path_lfp, overwrite=True)
            probe_df['channel_labels'] = channel_labels
            probe_df = pd.concat([probe_df, pd.DataFrame(channel_features)], axis=1)
            probe_df['probe_label'] = [probe_label] * channel_labels.shape[0]
            probe_df['eid'] = [eid] * channel_labels.shape[0]
            probe_df['pid'] = [pid] * channel_labels.shape[0]
            probe_df.to_pickle(file_path_probe)

            # Cleanup
            del lfpdata, lfp_mne, dsets, raw, bin_file
            sr_lf.close()
        else:
            print(f'LFP data already exists for {pid}')

    print("Extraction complete.")
    return 




