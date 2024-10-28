

import config 
from config import BRAIN_REGION_PAIRS
import submitit
import os
import gc
from eid_probe_info_table import eid_probe_info_table

##########################
# Submit jobs
##########################

# initialize a list in which our returning jobs will be stored
joblist=[]

# loop over array_parallel
print('#### Start submitting jobs #####')
jcount=0
for i, pair in enumerate(BRAIN_REGION_PAIRS):
    # executor is the submission interface (logs are dumped in the folder)
    log_dir  = config.log_dir
    executor = submitit.AutoExecutor(folder= log_dir)

    # set memory, timeout in min, and partition for running the job
    executor.update_parameters(**config.SUBMITIT_PARAMS)

    # actually submit the job: note that "value" correspond to that of array_parallel in this iteration
    job = executor.submit(eid_probe_info_table, pair)

    # add info about job submission order
    job.job_initial_indice=i 

    # print the ID of your job
    print("submit job" + str(job.job_id))  

    # append the job to the joblist
    joblist.append(job)

    # increase the job count
    jcount=jcount+1

### now that the loop has ended we check whether any job is already done
print('#### Start waiting for jobs to return #####')
njobs_finished = sum(job.done() for job in joblist)

# decide whether we clean our job live or not
clean_jobs_live= config.clean_jobs_live

# create a list to store finished jobs
finished_list=[]
finished_order=[]

## now we will keep looking for a new finished job until all jobs are done:
njobs_finished=0
while njobs_finished<jcount:
  doneIdx=-1
  for j, job in enumerate(joblist):
    if job.done():
      doneIdx=j
      break
  if doneIdx>=0:
    print(str(1+njobs_finished)+' on ' + str(jcount))
    # report last job finished
    print("last job finished: " + job.job_id)
    # obtain result from job
    job_result=job.result()
    # do some processing with this job
    print(job_result)
    # decide what to do with the finished job object
    if clean_jobs_live:
      # delete the job object
      del job
      # collect all the garbage immediately to spare memory
      gc.collect()
    else:
      # if we decided to keep the jobs in a list for further processing, add it finished job list 
      finished_list.append(job)
      finished_order.append(job.job_initial_indice)
    # increment the count of finished jobs
    njobs_finished=njobs_finished+1
    # remove this finished job from the initial joblist
    joblist.pop(doneIdx)
    
print('#### All jobs completed #####')  