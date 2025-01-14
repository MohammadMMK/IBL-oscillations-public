
def get_session_list_acronym(region, mode='load', minimum_number_of_channels=0, path=None, save=True):      
    """
    Retrieve or load a list of session information for a specified brain region.
    Parameters:
    region (str): The brain region acronym to filter sessions.
    mode (str): The mode of operation, either 'download' to fetch data from the server or 'load' to read from a local file. Default is 'load'.
    minimum_number_of_channels (int): The minimum number of channels required in the specified region. Default is 0.
    path (str): The file path to save or load the session information. If None, a default path is used. Default is None.
    save (bool): Whether to save the downloaded data to a CSV file. Only applicable if mode is 'download'. Default is True.
    Returns:
    pd.DataFrame: A DataFrame containing session information including session ID, probe ID, probe label, region, and channel count.
    Raises:
    ValueError: If an invalid mode is provided.
    """
    if mode == 'download':
        # Import libraries
        import pandas as pd
        import numpy as np
        from brainbox.io.one import load_channel_locations
        import os
        from one.api import ONE
        ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
        one = ONE(password='international')

        # Initialize a list to store the results
        eid_probe_info = []

        # Function to retrieve sessions for the specified region
        def get_sessions(region):
            session_list = one.alyx.rest(
                'sessions', 'list',
                task_protocol='ephys', performance_gte=70, 
                dataset_qc_gte='PASS', dataset='*lf*', 
                atlas_acronym=region
            )
            eids = [session['id'] for session in session_list]
            return eids

        # Get session IDs for the specified region
        eids = get_sessions(region)

        # Loop through sessions
        for eid in eids:
            # Find the probe labels for the specified region
            pid_list = [p['id'] for p in one.alyx.rest('insertions', 'list', session=eid, atlas_acronym=region)]
            probe_region = [one.pid2eid(p)[1] for p in pid_list]

            # Initialize a list to store channel counts
            channels_count = []

            # Load channels for each probe assigned to the region
            for i, probe in enumerate(probe_region):
                channels = load_channel_locations(eid, probe, one=one)[probe]
                channels_df = pd.DataFrame(channels)
                channels_filtered = channels_df[channels_df['acronym'].isin([f'{region}1', f'{region}2/3', f'{region}4', f'{region}5', f'{region}6a', f'{region}6b', f'{region}'])]
                channels_count = len(channels_filtered)
                if len(channels_count) > minimum_number_of_channels:
                    eid_probe_info.append((eid, pid_list[i]),probe_region, region,  channels_count)

        # Create a DataFrame to display the data
        df = pd.DataFrame(eid_probe_info, columns=['eid', 'pid', f'probe_label', 'region', 'channels_count'])
        if save:
            if path is None:
                path = os.path.join('data', f'eid_probe_info_{region}.csv')
            df.to_csv(path, index=False)

        return df
    
    if mode == 'load':
        if path is None:
            path = os.path.join('data', f'eid_probe_info_{region}.csv')
        return pd.read_csv(path)
    
    else:
        raise ValueError("Invalid mode. Choose either 'load' or 'download'.")



def get_session_list_pair(pair_region, mode = 'load', minimum_number_of_channels = 0, path = None, save = True):      
    """
    Retrieve or load session information for a pair of brain regions.
    Parameters:
    pair_region (tuple): A tuple containing two brain region acronyms (e.g., ('region1', 'region2')).
    mode (str): Mode of operation, either 'download' to fetch data from the server or 'load' to load from a local file. Default is 'load'.
    minimum_number_of_channels (int): Minimum number of channels required for a probe to be included. Default is 0.
    path (str): Path to save or load the CSV file. If None, a default path is used. Default is None.
    save (bool): Whether to save the downloaded data to a CSV file. Default is True.
    Returns:
    pd.DataFrame: A DataFrame containing session and probe information for the specified brain regions.
    Raises:
    FileNotFoundError: If the file is not found in 'load' mode and the path is incorrect or data has not been downloaded yet.
    """
   
    if mode == 'download':

        ############
        # Imports libraries
        import pandas as pd
        import numpy as np
        from brainbox.io.one import load_channel_locations
        import os
        from one.api import ONE
        ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
        one = ONE(password='international')

        # Initialize a list to store the results
        eid_probe_info = []
        # Function to retrieve sessions for a region
        def get_sessions(region):
            session_list = one.alyx.rest(
                'sessions', 'list',
                task_protocol='ephys', performance_gte=70, 
                dataset_qc_gte='PASS', dataset='*lf*', 
                atlas_acronym=region
            )
            eids = [session['id'] for session in session_list]
            return eids
        
        def get_probes(eid, region, minimum_number_of_channels = minimum_number_of_channels):
            pids = [p['id'] for p in one.alyx.rest('insertions', 'list', session=eid, atlas_acronym=region)]
            probes = [one.pid2eid(p)[1] for p in pids]
            channels_count = []
            # Load channels for each probe assigned to region 1
            for i, probe in enumerate(probes):
                channels = load_channel_locations(eid, probe, one=one)[probe]
                channels = pd.DataFrame(channels)
                channels = channels[channels['acronym'].isin([f'{pair_region[0]}1', f'{pair_region[0]}2/3', f'{pair_region[0]}4', f'{pair_region[0]}5', f'{pair_region[0]}6a', f'{pair_region[0]}6b', f'{pair_region[0]}'])]
                if len(channels) < minimum_number_of_channels:
                    pids.pop(i), probes.pop(i)
                else:
                    channels_count.append(len(channels))
            return pids, probes, channels_count

        # Get session IDs for both regions
        eids_1 = get_sessions(pair_region[0])
        eids_2 = get_sessions(pair_region[1])
        shared_eids = np.intersect1d(eids_1, eids_2)

        # Loop through shared sessions
        for eid in shared_eids:
            
            pids1, probes1, channels_count1  = get_probes(eid, pair_region[0])
            pids2, probes2, channels_count2  = get_probes(eid, pair_region[1])
            eid_probe_info.append((eid, pids1 , pids2, probes1, probes2, pair_region[0], pair_region[1], channels_count1, channels_count2))

        # Create a DataFrame to display the data
        df = pd.DataFrame(eid_probe_info, columns= ['eid', 'pid_1', 'pid_2', 'probe_1', 'probe_2', 'region_1', 'region_2', 'channels_count_1', 'channels_count_2'])
        if path is None:
            path = os.path.join('data', f'eid_probe_info_{pair_region[0]}_{pair_region[1]}.csv' )
        else:
            path = path
        if save:
            df.to_csv(path, index=False)

        return df
    
    if mode == 'load':
        if path is None:
            path = os.path.join('data', f'eid_probe_info_{pair_region[0]}_{pair_region[1]}.csv' )
        else:
            path = path
        if os.path.isfile(path):
            return pd.read_csv(path)
        else:
            raise FileNotFoundError(f"File not found: {path}, please download the data first")
        
    




