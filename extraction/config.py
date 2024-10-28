# config.py
import os

# 1) p
BRAIN_REGION_PAIRS = [('VISam', 'VISam')]  # for eid_probe_info_table.py you can for several pairs but for data_extraction.py you should put only one pair 
minimum_number_of_channels = 10  # minimum number of channels per region to consider for a session

# 2) 

extraction_parameters = {
    'extract_wheel': True,
    'extract_dlc': True,
    'extract_spikes': True,
    'extract_lfp': True,
    'overwrite_behavior': False,
    'overwrite_wheel': False,
    'overwrite_dlc': False,
    'overwrite_spikes': False,
    'overwrite_lfp': False,
    'resampled_fs': 500
}


# 3) Submitit parameters
log_dir = os.getcwd()+'/logs/'
clean_jobs_live = True
maxjobs = 60
SUBMITIT_PARAMS = {
    'slurm_partition': 'CPU',  
    'mem_gb': 24,
    'timeout_min': 200,  
    'cpus_per_task': 1, 
    'slurm_array_parallelism': maxjobs}

