import pandas as pd
from pathlib import Path
from one.api import ONE

# One-time setup for ONE with cache mode set to 'remote'
one = ONE(base_url='https://openalyx.internationalbrainlab.org')

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

def get_spikes(pid, modee='download', path=None):
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
