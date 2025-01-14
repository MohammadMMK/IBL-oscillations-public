import pandas as pd
from pathlib import Path
from one.api import ONE
import os
from .epoching import numpyToMNE
import sys
sys.path.append(str(Path(os.getcwd()).resolve().parent)) # add the root of the project to the python path
# Now import the paths from config.py
from config import paths
# One-time setup for ONE with cache mode set to 'remote'
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', cache_dir = paths['cache_dir'],  silent=True)
one = ONE(password='international')


def get_behavior(eid, modee='load', path=None):
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

def get_spikes(pid, modee='load', path=None):
    """
    Retrieve spike data for a given probe insertion ID (pid).
    Parameters:
    pid (str): Probe insertion ID.
    modee (str): modee of operation, either 'load' to load from a file or 'download' to fetch from the IBL database. Default is 'load'.
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

def get_channels(eid, pid, modee='load', path=None):
    """
    Retrieve channel data for a given probe insertion ID (pid).
    Parameters:
    pid (str): Probe insertion ID.
    modee (str): modee of operation, either 'load' to load from a file or 'download' to fetch from the IBL database. Default is 'load'.
    path (str or None): Path to the file containing channel data. If None and modee is 'load', a default path is constructed. Default is None.
    Returns:
    pd.DataFrame: DataFrame containing channel data.
    Raises:
    FileNotFoundError: If the file specified in 'path' does not exist when modee is 'load'.
    ValueError: If an invalid modee is provided.
    """
    if modee == 'load':
        if path is None:
            path = Path(f'/mnt/data/AdaptiveControl/IBLrawdata/eid_data/{eid}/probe_{pid}.pkl')
        else:
            path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_pickle(path)
    
    elif modee == 'download':
        from brainbox.io.one import SpikeSortingLoader
        ssl = SpikeSortingLoader(pid=pid, one=one)
        channels_df = pd.DataFrame(ssl.load_channels())
        return channels_df
    
    else:
        raise ValueError("Invalid modee. Choose either 'load' or 'download'.")

from one.api import ONE
import brainbox.task.passive as passive
from brainbox.io.one import SpikeSortingLoader
import ibldsp.voltage as voltage
import spikeglx
import mne 
import numpy as np
import pandas as pd
from scipy.stats import skew
import mne 
# add path to import the module
import sys 
import os
from .detect_bad_channels import detect_bad_channels

def get_epoch_StimOn(pid, modee = 'download', window_secs=[-1, 1.5], save = True, ephys_path = '/mnt/data/AdaptiveControl/IBLrawdata/LFP/', overwrite=False):
    import os 
    import numpy as np
    file_path_epochs = os.path.join(ephys_path, f'{pid}_epoch_stimOn.npy')
    if modee == 'load':
        if not os.path.exists(file_path_epochs):
            raise FileNotFoundError(f"File not found: {file_path_epochs}")
        epochs = np.load(file_path_epochs)
        return epochs
    elif modee == 'download':

        if os.path.exists(file_path_epochs) and not overwrite:
            print(f' file exist for {file_path_epochs}...')
            return np.load(file_path_epochs)
        
        ssl = SpikeSortingLoader(pid=pid, one=one)
        eid, probe_label = one.pid2eid(pid)
        dsets = one.list_datasets(eid, collection=f'raw_ephys_data/{probe_label}', filename='*.lf.*')
        data_files, info = one.load_datasets(eid, dsets, download_only=True)
        bin_file = next(df for df in data_files if df.suffix == '.cbin')
        sr_lf = spikeglx.Reader(bin_file)
        print(sr_lf.fs)
        channels = ssl.load_channels()
        channels_name = pd.DataFrame(channels)['acronym'].values
        trials = one.load_object(ssl.eid, 'trials', collection='alf')
        trials = trials.to_df()
        behavior_stimOnset = trials.dropna(subset=['stimOn_times'])
        behavior_stimOnset = behavior_stimOnset.reset_index(drop=True)
        stimOn_times = behavior_stimOnset['stimOn_times'].values

        import numpy as np
        import os

        # Define the shape of the data (number of epochs, number of channels, number of samples)
        num_epochs = len(stimOn_times)
        num_channels = 384
        num_samples = int((window_secs[1] - window_secs[0]) * 2500 / 5)  # Adjust based on decimation

        # Create a memory-mapped file
        epochs_data_np = np.memmap('epochs_data.dat', dtype='float32', mode='w+', shape=(num_epochs, num_channels, num_samples))

        for i in range(len(stimOn_times)):
            print(f'Epoch {i+1}/{len(stimOn_times)}')
            t_event = stimOn_times[i]
            start_time = t_event + window_secs[0]
            last_time = t_event + window_secs[1]
            first, last = ssl.samples2times([start_time, last_time], direction='reverse') / 12
            tsel = slice(int(first), int(last))
            raw_lf = sr_lf[tsel, :-sr_lf.nsync].T

            # Destripe and decimate LFP data
            destriped = voltage.destripe_lfp(raw_lf.astype(float), fs=sr_lf.fs)
            decimated = mne.filter.resample(destriped.astype(float), up=2.0, down=10.0, 
                                            window='boxcar', npad='auto', pad='reflect_limited', verbose=False)

            # Store the decimated data in the memory-mapped array
            epochs_data_np[i, :, :] = decimated
            del raw_lf, destriped, decimated

        # Flush changes to disk
        epochs_data_np.flush()
        if save == True:
            np.save(file_path_epochs, epochs_data_np)
            return 0
        else:
            return epochs_data_np


import numpy as np
from one.api import ONE
import json

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
