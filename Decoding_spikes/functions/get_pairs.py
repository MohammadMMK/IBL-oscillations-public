import numpy as np
from one.api import ONE
import json

def visual_pid_eid_pairs(output_file='pid_eid_pairs.json', only_passive=True):

    # Initialize ONE with the IBL public server
    one = ONE(base_url='https://openalyx.internationalbrainlab.org')

    # Define the visual areas of interest
    visual_areas = ['VISp', 'VISpm', 'VISam', 'VISa', 'VISrl', 'VISal', 'VISli', 'VISl']

    # Retrieve all probe insertions for the specified visual areas
    insertions = []
    for area in visual_areas:
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
    print(f"Total number of probe insertions: {len(pid_eid_pairs)}")

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
    

    # Save the pairs to a JSON file
    with open(output_file, 'w') as f:
        json.dump(pid_eid_pairs, f)
    
    print(f"PID-EID pairs saved to {output_file}")
    return pid_eid_pairs
