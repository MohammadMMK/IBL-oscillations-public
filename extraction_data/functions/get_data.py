import pandas as pd
from pathlib import Path
from one.api import ONE
import os
from .epoching import numpyToMNE
from .filter_trials import filter_trials
import sys

# One-time setup for ONE with cache mode set to 'remote'
one = ONE(base_url='https://openalyx.internationalbrainlab.org', cache_rest=None, mode='remote')
one = ONE(password='international')
from datetime import timedelta
from config import LFP_mode, LFP_dir, REST_cache_expirry_min

one.alyx.default_expiry = timedelta(minutes= REST_cache_expirry_min)



from brainbox.io.one import SpikeSortingLoader
import ibldsp.voltage as voltage
import spikeglx
import numpy as np
import mne 
import sys 
import json


def get_behavior(eid, modee='download', path=None):
    """
    Retrieve behavioral data for a given experiment ID (eid).
    Parameters:
    eid (str): The experiment ID for which to retrieve behavioral data.
    modee (str, optional): modee of operation, either 'load' to load from a local file or 'download' to fetch from the IBL database. Default is 'load'.
    path (str or Path, optional): Path to the local file to load. If not provided, a default path is constructed based on the eid.
    Returns:
    pd.DataFrame: A DataFrame containing the behavioral data.
    Raises:
    FileNotFoundError: If the file specified in 'load' modee does not exist.
    ValueError: If an invalid modee is provided.
    """
    if modee == 'load':
        if path is None:
            path = Path(f'/mnt/data/AdaptiveControl/IBLrawdata/eid_data/{eid}/trials_{eid}.pkl')
        else:
            path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_pickle(path)
    
    elif modee == 'download':
        behavior = one.load_object(eid, 'trials', collection='alf')
        return behavior.to_df()
    
    else:
        raise ValueError("Invalid modee. Choose either 'load' or 'download'.")

def get_spikes(pid, modee='download', path=None):
    """
    Retrieve spike data for a given probe insertion ID (pid).
    Parameters:
    pid (str): Probe insertion ID.
    modee (str): modee of operation, either 'load' to load from a file or 'download' to fetch from the IBL database. Default is 'download'.
    path (str or None): Path to the file containing spike data. If None and modee is 'load', a default path is constructed. Default is None.
    Returns:
    pd.DataFrame: DataFrame containing spike data.
    Raises:
    FileNotFoundError: If the file specified in 'path' does not exist when modee is 'load'.
    ValueError: If an invalid modee is provided.
    """
    if modee == 'load':
        if path is None:
            path = Path(f'/mnt/data/AdaptiveControl/IBLrawdata/pid_data/{pid}/spikes_{pid}.pkl')
        else:
            path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_pickle(path)
    
    elif modee == 'download':
        from brainbox.io.one import SpikeSortingLoader
        ssl = SpikeSortingLoader(pid=pid, one=one)
        spike_spikes, spike_clusters, _ = ssl.load_spike_sorting()
        spikes_df = pd.DataFrame({'spikes': spike_spikes, 'clusters': spike_clusters})
        return spikes_df
    
    else:
        raise ValueError("Invalid modee. Choose either 'load' or 'download'.")

def get_channels(pid):

    from brainbox.io.one import SpikeSortingLoader
    ssl = SpikeSortingLoader(pid=pid, one=one)
    channels_df = pd.DataFrame(ssl.load_channels())
    return channels_df



def get_epoch_StimOn(
    pid, LFP_mode= None, **kwargs
):
 
    tmin = kwargs.get('tmin', -1)
    tmax = kwargs.get('tmax', 1)
    save = kwargs.get('save', True)
    overwrite = kwargs.get('overwrite', False)
    contrasts = kwargs.get('contrasts', "all")
    stim_side = kwargs.get('stim_side', "Both")
    prob_left = kwargs.get('prob_left', 'all')
    remove_first_trials_of_block = kwargs.get('remove_first_trials_of_block', False)

    from config import  LFP_dir
    if LFP_mode is None:
        from config import LFP_mode
    ## File paths
    file_path_epochs = os.path.join(LFP_dir, f"{pid}_epoch_stimOn.fif")
    file_path_memmap = f"temp_{pid}.dat"
    window_secs=[tmin, tmax]
    try:
        ## ------------------------ MODE: LOAD EXISTING EPOCHS ------------------------ ##
        if LFP_mode == "load":
            if not os.path.exists(file_path_epochs):
                raise FileNotFoundError(f"File not found: {file_path_epochs}")
            
            epochs = mne.read_epochs(file_path_epochs)

            ## Filter loaded epochs based on metadata
            if epochs.metadata is not None:
                metadata_filtered = filter_trials(
                    epochs.metadata, prob_left, contrasts, stim_side, remove_first_trials_of_block
                )

                # Select only the filtered epochs
                selected_indices = metadata_filtered.index.to_numpy()
                epochs = epochs[selected_indices]
                epochs.metadata = metadata_filtered  # Update metadata
            else:
                print("No metadata found for the epochs. Skipping filtering.")
            
            # trim the epochs
            epochs = epochs.crop(tmin=tmin, tmax=tmax)
                
            return epochs

        ## ------------------------ MODE: DOWNLOAD AND PROCESS DATA ------------------------ ##
        elif LFP_mode == "download":
            # Check if the file already exists
            if os.path.exists(file_path_epochs) and not overwrite:
                print(f"File exists for {file_path_epochs}... Loading saved epochs.")
                return mne.read_epochs(file_path_epochs)

            ## Step 1: Download Ephys Data
            ssl = SpikeSortingLoader(pid=pid, one=one)
            eid, probe_label = one.pid2eid(pid)
            dsets = one.list_datasets(eid, collection=f"raw_ephys_data/{probe_label}", filename="*.lf.*")
            data_files, info = one.load_datasets(eid, dsets, download_only=True)
            bin_file = next(df for df in data_files if df.suffix == ".cbin")
            sr_lf = spikeglx.Reader(bin_file)
            print(f"Raw sampling rate: {sr_lf.fs}")

            ## Step 2: Load Channel Information
            channels = ssl.load_channels()
            channels_name = pd.DataFrame(channels)["acronym"].values
            ch_names = [f"{ch}_{i}" for i, ch in enumerate(channels_name)]
            
            ## Step 3: Load and Filter Trials
            trials = get_behavior(eid, modee="download")
            behavior_stimOnset = trials.dropna(subset=["stimOn_times"]).reset_index(drop=True)

            ## ------------------------ FILTER TRIALS ------------------------ ##
            behavior_stimOnset = filter_trials(behavior_stimOnset, prob_left, contrasts, stim_side, remove_first_trials_of_block)

            ## Extract final filtered stimulus onset times
            stimOn_times = behavior_stimOnset["stimOn_times"].values

            ## ------------------------ PROCESS LFP DATA ------------------------ ##
            num_epochs = len(stimOn_times)
            num_channels = 384
            num_samples = int((window_secs[1] - window_secs[0]) * 2500 / 5)  # Adjust based on decimation

            # Create a memory-mapped file
            epochs_data_np = np.memmap(
                file_path_memmap, dtype="float32", mode="w+", shape=(num_epochs, num_channels, num_samples)
            )
            
            ## Loop through each trial and extract LFP
            for i in range(num_epochs):
                if i % 100 == 0:
                    print(f"Processing epoch {i+1}/{num_epochs}")
                
                t_event = stimOn_times[i]
                start_time = t_event + window_secs[0]
                last_time = t_event + window_secs[1]
                first, last = ssl.samples2times([start_time, last_time], direction="reverse") / 12
                tsel = slice(int(first), int(last))
                
                raw_lf = sr_lf[tsel, :-sr_lf.nsync].T

                ## Destripe and Decimate LFP Data
                destriped = voltage.destripe_lfp(raw_lf.astype(float), fs=sr_lf.fs)
                decimated = mne.filter.resample(
                    destriped.astype(float), up=2.0, down=10.0, 
                    window="boxcar", npad="auto", pad="reflect_limited", verbose=False
                )

                epochs_data_np[i, :, :] = decimated

                # Clean up
                del raw_lf, destriped, decimated

            ## ------------------------ CONVERT TO MNE EPOCHS ------------------------ ##
            epochs_mne = numpyToMNE(epochs_data_np, behavior_stimOnset, ch_names, sfreq=500)
            del epochs_data_np

            ## Save or return epochs
            if save:
                epochs_mne.save(file_path_epochs, overwrite=True)
                return 0
            else:
                return epochs_mne
    
    finally:
        ## ------------------------ CLEAN UP TEMP FILES ------------------------ ##
        if os.path.exists(file_path_memmap):
            os.remove(file_path_memmap)
            print(f"File {file_path_memmap} has been deleted.")






def get_pid_eid_pairs(output_file= None, only_passive=True, regions = ['VISp', 'VISpm', 'VISam', 'VISa', 'VISrl', 'VISal', 'VISli', 'VISl']):

    # Initialize ONE with the IBL public server
    # Define the visual areas of interest

    # Retrieve all probe insertions for the specified visual areas
    insertions = []
    for area in regions:
        insertions += one.alyx.rest(
            'insertions', 'list', 
            task_protocol='ephys', 
            performance_gte=70, 
            dataset_qc_gte='PASS', 
            atlas_acronym=area
        )

    # Remove duplicates by insertion ID
    insertions = {insertion['id']: insertion for insertion in insertions}.values()

    # Create pairs of probe insertion ID and session ID
    pid_eid_pairs = [(insertion['id'], insertion['session']) for insertion in insertions]
    print(f"Total number of possible probe insertions: {len(pid_eid_pairs)}")

    if only_passive:
        # Filter out sessions without passive data
        eids = [eid for _, eid in pid_eid_pairs]
        no_passive_eids = []
        for eid in eids:
            datasets = one.list_datasets(eid)
            datasets_passive = [dataset for dataset in datasets if 'passiveGabor' in dataset]
            if len(datasets_passive) == 0:
                no_passive_eids.append(eid)
        print(f"Number of sessions without passive data: {len(no_passive_eids)}")

        # Keep only sessions with passive data
        eid_with_passive = [eid for eid in eids if eid not in no_passive_eids]
        pid_eid_pairs = [(pid, eid) for pid, eid in pid_eid_pairs if eid in eid_with_passive]
        print(f"Number of sessions with passive data: {len(pid_eid_pairs)}")
    
    if output_file:
        # Save the pairs to a JSON file
        with open(output_file, 'w') as f:
            json.dump(pid_eid_pairs, f)
        print(f"PID-EID pairs saved to {output_file}")
    return pid_eid_pairs
