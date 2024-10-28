import pandas as pd
from pathlib import Path

def get_behavior(eid, mode='load', path=None):
    """
    Retrieve behavioral data for a given experiment ID (eid).
    Parameters:
    eid (str): The experiment ID for which to retrieve behavioral data.
    mode (str, optional): Mode of operation, either 'load' to load from a local file or 'download' to fetch from the IBL database. Default is 'load'.
    path (str or Path, optional): Path to the local file to load. If not provided, a default path is constructed based on the eid.
    Returns:
    pd.DataFrame: A DataFrame containing the behavioral data.
    Raises:
    FileNotFoundError: If the file specified in 'load' mode does not exist.
    ImportError: If the 'ONE' API is not installed when 'download' mode is selected.
    ValueError: If an invalid mode is provided.
    """
    if mode == 'load':
        # Set default path if none provided
        if path is None:
            path = Path(f'/mnt/data/AdaptiveControl/IBLrawdata/eid_data/{eid}/trials_{eid}.pkl')
        else:
            path = Path(path)

        # Ensure file exists
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        
        return pd.read_pickle(path)
    
    elif mode == 'download':
        try:
            from one.api import ONE
            ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
            one = ONE(password='international')
            behavior = one.load_object(eid, 'trials', collection='alf')
            return behavior.to_df()
        except ImportError:
            raise ImportError("Failed to import `ONE`. Please ensure the IBL ONE API is installed.")

    else:
        raise ValueError("Invalid mode. Choose either 'load' or 'download'.")

def get_spikes(pid, mode='load', path=None):
    """
    Retrieve spike data for a given probe insertion ID (pid).
    Parameters:
    pid (str): Probe insertion ID.
    mode (str): Mode of operation, either 'load' to load from a file or 'download' to fetch from the IBL database. Default is 'load'.
    path (str or None): Path to the file containing spike data. If None and mode is 'load', a default path is constructed. Default is None.
    Returns:
    pd.DataFrame: DataFrame containing spike data.
    Raises:
    FileNotFoundError: If the file specified in 'path' does not exist when mode is 'load'.
    ImportError: If the 'ONE' module cannot be imported when mode is 'download'.
    ValueError: If an invalid mode is provided.
    """
    if mode == 'load':
        # Set default path if none provided
        if path is None:
            path = Path(f'/mnt/data/AdaptiveControl/IBLrawdata/pid_data/{pid}/spikes_{pid}.pkl')
        else:
            path = Path(path)

        # Ensure file exists
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        
        return pd.read_pickle(path)
    
    elif mode == 'download':
        try:
            from one.api import ONE
            ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
            one = ONE(password='international')
            from brainbox.io.one import SpikeSortingLoader
            ssl = SpikeSortingLoader(pid=pid, one=one)
            spike_spikes, spike_clusters, spike_channels = ssl.load_spike_sorting()
            spikes_df = pd.DataFrame({'spikes': spike_spikes, 'clusters': spike_clusters})
            return spikes_df
        except ImportError:
            raise ImportError("Failed to import `ONE`. Please ensure the IBL ONE API is installed.")

    else:
        raise ValueError("Invalid mode. Choose either 'load' or 'download'.")
    

def get_channels(eid, pid, mode='load', path=None):
    """
    Retrieve channel data for a given probe insertion ID (pid).
    Parameters:
    pid (str): Probe insertion ID.
    mode (str): Mode of operation, either 'load' to load from a file or 'download' to fetch from the IBL database. Default is 'load'.
    path (str or None): Path to the file containing channel data. If None and mode is 'load', a default path is constructed. Default is None.
    Returns:
    pd.DataFrame: DataFrame containing channel data.
    Raises:
    FileNotFoundError: If the file specified in 'path' does not exist when mode is 'load'.
    ImportError: If the 'ONE' module cannot be imported when mode is 'download'.
    ValueError: If an invalid mode is provided.
    """
    if mode == 'load':
        # Set default path if none provided
        if path is None:
            path = Path(f'/mnt/data/AdaptiveControl/IBLrawdata/eid_data/{eid}/probe_{pid}.pkl')
        else:
            path = Path(path)

        # Ensure file exists
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        
        return pd.read_pickle(path)
    
    elif mode == 'download':
        try:
            from one.api import ONE
            ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
            one = ONE(password='international')
            from brainbox.io.one import SpikeSortingLoader
            ssl = SpikeSortingLoader(pid=pid, one=one)
            channels_df = pd.DataFrame(ssl.load_channels())
            return channels_df
        except ImportError:
            raise ImportError("Failed to import `ONE`. Please ensure the IBL ONE API is installed.")

    else:
        raise ValueError("Invalid mode. Choose either 'load' or 'download'.")