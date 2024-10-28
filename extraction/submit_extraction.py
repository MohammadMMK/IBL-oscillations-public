import os 
from config import  BRAIN_REGION_PAIRS, SUBMITIT_PARAMS, extraction_parameters
import pandas as pd
import numpy as np
import mne
import submitit
import gc
import config
import time
from functools import partial

from data_extraction import data_extraction


data_extraction_with_params = partial(data_extraction, **extraction_parameters)
##########################
# Submit jobs
##########################

if len(BRAIN_REGION_PAIRS) >1 :
    raise ValueError("BRAIN_REGION_PAIRS should have only one pair of brain regions for extraction")

region1 = BRAIN_REGION_PAIRS[0][0]
region2 = BRAIN_REGION_PAIRS[0][1]

table_path = os.path.join(os.getcwd(), f'data/eid_probe_info_{region1}_{region2}.csv')
if not os.path.isfile(table_path):
    print(f"eid_probe_info_{region1}_{region2}.csv does not exist at {os.getcwd()}")
    print("use eid_probe_info_table.py to generate the table")
    raise ValueError(f"table does not exist at {os.getcwd()}")

# Load the table
df = pd.read_csv(table_path)
job_args = []

for i, row in df.iterrows():
    # if i>0:
    #     break
    eid = row['eid']
    probe_label_1 = row[f'label_{region1}'][2:-2]
    probe_label_2 = row[f'label_{region2}'][2:-2]
    probe_labels = [probe_label_1, probe_label_2]
    probe_labels = list(set(probe_labels))
    

    for probe_label in probe_labels:
        job_args.append((eid, probe_label))


print(f'Number of jobs to submit: {len(job_args)}')
log_dir  = config.log_dir
max_jobs_parallel = config.maxjobs
print(f'Maximum number of jobs to run in parallel: {max_jobs_parallel}')
executor = submitit.AutoExecutor(folder=log_dir)
executor.update_parameters(**config.SUBMITIT_PARAMS)


jobs = executor.map_array(data_extraction_with_params, [eid for eid, probe_label in job_args], [probe_label for eid, probe_label in job_args])
    

# # Start monitoring job progress
# print(f'Submitted {len(jobs)} jobs. Waiting for completion...')

# njobs_finished = 0
# total_jobs = len(jobs)

# while njobs_finished < total_jobs:
#     # Count how many jobs have finished
#     finished_jobs = sum(job.done() for job in jobs)
    
#     # Update the finished count if more jobs have completed since the last check
#     if finished_jobs > njobs_finished:
#         print(f'{finished_jobs}/{total_jobs} jobs finished.')
#         njobs_finished = finished_jobs
    
#     # Sleep briefly before checking again
#     time.sleep(10)

# print('All jobs have completed.')