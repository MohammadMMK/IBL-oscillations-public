def monitor_job_status(all_jobs):
    # Monitor job status
    total_jobs = len(all_jobs)
    print(f"Submitted {total_jobs} jobs.")
    finished_jobs = 0
    completed_jobs_set = set()
    while finished_jobs < total_jobs:
        for idx, job in enumerate(all_jobs):
            if job.done() and idx not in completed_jobs_set:
                finished_jobs += 1
                completed_jobs_set.add(idx)
                print(f"Jobs finished: {finished_jobs}/{total_jobs}")
                # remove the job from the list
                all_jobs.pop(idx)
    print("All jobs are finished.")