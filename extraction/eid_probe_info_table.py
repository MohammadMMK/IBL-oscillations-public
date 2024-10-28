
import config 

def eid_probe_info_table(pair_region):
    """
    Generates a table containing session IDs and the count of channels for specified brain regions.
    Parameters:
    pair_region (tuple):  containing two brain region acronyms for example ('VISp', 'VISl')
    Returns:
    pd.DataFrame: A DataFrame with columns 'eid', the first brain region acronym, and the second brain region acronym.
                  Each row contains the session ID and dictionaries with the count of channels per probe for each region.
    """
    ############
    # Imports libraries
    import pandas as pd
    import numpy as np
    from brainbox.io.one import load_channel_locations
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

    # Get session IDs for both regions
    eids_1 = get_sessions(pair_region[0])
    eids_2 = get_sessions(pair_region[1])
    shared_eids = np.intersect1d(eids_1, eids_2)

    # Loop through shared sessions
    for eid in shared_eids:
        # Find the probe label for the first region
        pid1 = [p['id'] for p in one.alyx.rest('insertions', 'list', session=eid, atlas_acronym=pair_region[0])]
        probe_region_1 = [one.pid2eid(p)[1] for p in pid1]


        # Initialize a dictionary to store channel counts for region 1
        channels_count_region1 = []
        
        # Load channels for each probe assigned to region 1
        for probe in probe_region_1:
            channels_region1 = load_channel_locations(eid, probe, one=one)[probe]
            channels_1 = pd.DataFrame(channels_region1)
            channels_1 = channels_1[channels_1['acronym'].isin([f'{pair_region[0]}1', f'{pair_region[0]}2/3', f'{pair_region[0]}4', f'{pair_region[0]}5', f'{pair_region[0]}6a', f'{pair_region[0]}6b', f'{pair_region[0]}'])]
            # Store the count of channels per probe
            channels_count_region1.append(len(channels_1)) 

        # Find the probe label for the second region
        pid2 = [p['id'] for p in one.alyx.rest('insertions', 'list', session=eid, atlas_acronym=pair_region[1])]
        probe_region_2 = [one.pid2eid(p)[1] for p in pid1]

        # Initialize a dictionary to store channel counts for region 2
        channels_count_region2 = []
        
        # Load channels for each probe assigned to region 2
        for probe in probe_region_2:
            channels_region2 = load_channel_locations(eid, probe, one=one)[probe]
            channels_2 = pd.DataFrame(channels_region2)
            channels_2 = channels_2[channels_2['acronym'].isin([f'{pair_region[1]}1', f'{pair_region[1]}2/3', f'{pair_region[1]}4', f'{pair_region[1]}5', f'{pair_region[1]}6a', f'{pair_region[1]}6b', f'{pair_region[1]}'])]
            
            # Store the count of channels per probe
            channels_count_region2.append(len(channels_2))

        # Filter sessions where both regions have minimum amount of channels
        if sum(channels_count_region1) >= config.minimum_number_of_channels and sum(channels_count_region2) >= config.minimum_number_of_channels:
          # Append the results, including channel counts
          eid_probe_info.append((eid,  probe_region_1, probe_region_2, channels_count_region1, channels_count_region2, pid1, pid2))

    # Create a DataFrame to display the data
    df = pd.DataFrame(eid_probe_info, columns=['eid',f'label_{pair_region[0]}', f'label_{pair_region[1]}', f'nb_channel_{pair_region[0]}', f'nb_channel_{pair_region[1]}', 'pid1', 'pid2'])
    df.to_csv(f'data/eid_probe_info_{pair_region[0]}_{pair_region[1]}.csv', index=False)
    return df



  
