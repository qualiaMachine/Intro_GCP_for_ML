---
title: "Resource Management Best Practices on CHTC"
teaching: 25
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions

- How do I check my disk usage and job history on CHTC?
- What tools does HTCondor provide for monitoring and managing running jobs?
- How do I right-size my resource requests so jobs start faster?
- What are CHTC's runtime limits and how do I handle long-running jobs?

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Check disk usage and quota on the submit node using `du` and quota commands.
- Monitor, inspect, and remove HTCondor jobs with `condor_q`, `condor_watch_q`, and `condor_rm`.
- Use job ClassAds and log files to right-size CPU, memory, and GPU requests.
- Identify CHTC's runtime categories and know when checkpointing is needed.
- Apply an end-of-session cleanup checklist to be a good citizen on shared resources.

::::::::::::::::::::::::::::::::::::::::::::::::::

CHTC is free for UW-Madison researchers — there are no billing accounts and no surprise invoices. But "free" does not mean "unlimited." CHTC is a **shared resource**: every CPU, GPU, and gigabyte of storage you hold is unavailable to another researcher. Being a good citizen means requesting only what you need, monitoring your jobs, and cleaning up when you are done.

This episode covers the practical tools and habits that keep your work running smoothly and the cluster healthy for everyone.


## Checking your disk usage

Your `/home` directory has a quota (typically ~20 GB). If you exceed it, jobs may fail to write output and new jobs will not submit. Check your usage regularly:

```bash
# How much space am I using in my home directory?
du -sh /home/$USER

# Break it down by subdirectory (top-level only)
du -sh /home/$USER/*/
```

To check your quota and how close you are to the limit:

```bash
quota -vs
```

:::::::::::::::::::::::::::::::::::::: callout

#### Storage tiers as a reminder

| Location | Purpose | Typical quota |
|----------|---------|---------------|
| `/home/$USER` | Code, submit files, small data | ~20 GB |
| `/staging/$USER` | Large input/output files for jobs | ~500 GB (by request) |
| SQUID (`/squid/$USER`) | Publicly readable large files | ~100 GB (by request) |

Large datasets and model checkpoints should go in `/staging`, not `/home`. See the [CHTC file system guide](https://chtc.cs.wisc.edu/uw-research-computing/file-avail-largedata) for details.

::::::::::::::::::::::::::::::::::::::::::::::::::


## Checking your job history

After jobs complete (or fail), use `condor_history` to review what happened:

```bash
# Your recent jobs (most recent first)
condor_history $USER

# Show specific columns: job ID, status, runtime, memory used
condor_history $USER -af ClusterId JobStatus RemoteWallClockTime MemoryUsage
```

The `JobStatus` field uses numeric codes: **1** = Idle, **2** = Running, **3** = Removed, **4** = Completed, **5** = Held.

This is especially useful for debugging: if a job completed but produced bad output, you can check how long it ran and how much memory it actually used.


## Monitoring running jobs

### condor_q — your primary dashboard

```bash
# All your jobs in the queue
condor_q

# Just your jobs (explicit)
condor_q $USER

# Show only running jobs
condor_q -running

# Show only held jobs (these need attention!)
condor_q -held
```

### condor_watch_q — live updating view

`condor_watch_q` refreshes automatically, like `top` for your jobs:

```bash
condor_watch_q
```

Press `Ctrl+C` to exit. This is handy when you are waiting for jobs to start or watching a batch complete.

### Checking for held jobs

Held jobs are stuck and will not run until you fix the problem. Common causes include exceeded disk or memory requests, missing input files, and Docker image pull failures.

```bash
# See why jobs are held
condor_q -held

# More detail on a specific held job
condor_q JOB_ID -af HoldReason
```

:::::::::::::::::::::::::::::::::::::: callout

#### Do not ignore held jobs

Held jobs sit in the queue consuming your job slot allowance but doing no useful work. Check for them regularly and either fix the issue and release them (`condor_release JOB_ID`) or remove them (`condor_rm JOB_ID`).

::::::::::::::::::::::::::::::::::::::::::::::::::


## Removing jobs

```bash
# Remove a single job
condor_rm JOB_ID

# Remove all your jobs (careful!)
condor_rm $USER

# Remove a specific cluster of jobs
condor_rm CLUSTER_ID
```

Use `condor_rm` when a job is stuck, when you realize you submitted with wrong parameters, or when you need to free up your queue slots.


## Understanding job ClassAds and resource usage

Every HTCondor job carries a set of **ClassAds** — key-value attributes that describe the job's requests and actual usage. These are your best tool for right-sizing future jobs.

```bash
# Show ALL ClassAds for a job (verbose)
condor_q JOB_ID -l

# Show specific attributes (auto-format)
condor_q JOB_ID -af RequestCpus RequestMemory RequestDisk

# Check actual usage of a running job
condor_q JOB_ID -af MemoryUsage DiskUsage RemoteWallClockTime
```

After a job completes, use `condor_history` with the same `-af` flags:

```bash
condor_history JOB_ID -af RemoteWallClockTime MemoryUsage DiskUsage_RAW RequestMemory RequestCpus
```

Comparing **requested** resources to **actual** usage tells you whether to adjust your submit file for the next run.


## Right-sizing resource requests

This is one of the most impactful things you can do as a CHTC user. Over-requesting resources does not make your job run faster — it makes your job **wait longer** in the queue, because the scheduler has to find a machine with all the resources you asked for.

### The right-sizing workflow

1. **Start with a reasonable estimate** for your first run.
2. **Check actual usage** after the job completes (see ClassAds above, or check the job log file).
3. **Adjust your submit file** for the next run — request ~20% more than the actual usage as a safety margin.

### What to look for in job logs

Every HTCondor job writes a log file (specified by the `log` line in your submit file). At the end of a completed job, HTCondor appends a summary like:

```
Partitionable Resources :    Usage  Request Allocated
   Cpus                 :                 4         4
   Disk (KB)            :    35000  1048576   4184124
   Memory (MB)          :      870     8192      8192
```

In this example, the job requested 8192 MB of memory but only used 870 MB. You could safely request `1024` or `1500` MB next time, which would let the job match to more machines and start sooner.

### Common over-requesting mistakes

| Resource | Mistake | Better approach |
|----------|---------|-----------------|
| **CPUs** | Requesting 8 CPUs for a single-threaded Python script | Request 1 CPU unless your code uses multiprocessing/threading |
| **Memory** | Requesting 32 GB "just in case" | Check actual usage, request actual + 20% buffer |
| **GPUs** | Requesting 2 GPUs for code that uses only 1 | Only request multiple GPUs if your code explicitly supports multi-GPU |
| **Disk** | Requesting 100 GB when output is 2 GB | Check output sizes from a test run, add buffer |

::::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Right-size a request

A colleague shows you their submit file:

```
request_cpus = 8
request_memory = 16GB
request_disk = 50GB
request_gpus = 1
```

Their job log shows this usage summary:

```
Partitionable Resources :    Usage  Request Allocated
   Cpus                 :                 8         8
   Disk (KB)            :   524288 52428800  52428800
   Gpus                 :                 1         1
   Memory (MB)          :     2048    16384     16384
```

Their Python training script uses PyTorch with a single `model.to("cuda")` call and no `DataParallel` or multiprocessing. What changes would you suggest to their submit file?

::::::::::::::::: solution

The job used only 2048 MB (~2 GB) of memory out of 16 GB requested, and the disk usage was ~512 MB out of 50 GB. Since the script is single-threaded Python (no multiprocessing) and uses a single GPU, the CPUs are also over-requested. Suggested changes:

```
request_cpus = 1
request_memory = 3GB
request_disk = 1GB
request_gpus = 1
```

This requests ~50% more than actual usage for memory and disk as a safety margin, drops CPUs to 1 since the script is single-threaded, and keeps the single GPU. The job will match to far more machines and likely start much sooner.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::


## Storage cleanup

After jobs complete, clean up files you no longer need — especially in `/staging`:

```bash
# Check staging usage
du -sh /staging/$USER

# List what's in staging
ls -lh /staging/$USER/

# Remove old output directories
rm -rf /staging/$USER/old_experiment_output/

# Clean up large files in home
find /home/$USER -name "*.tar.gz" -size +100M -ls
```

:::::::::::::::::::::::::::::::::::::: callout

#### Make cleanup part of your workflow

A good habit is to add cleanup steps to your workflow: after you have copied important results to your local machine or a permanent archive, delete the copies on CHTC. The `/staging` filesystem is meant for active jobs, not long-term storage.

::::::::::::::::::::::::::::::::::::::::::::::::::


## CHTC's job runtime limits

CHTC jobs are categorized by maximum runtime. You declare this in your submit file so HTCondor can schedule your job onto an appropriate machine:

| Category | Max runtime | Submit file setting |
|----------|------------|---------------------|
| Short | 12 hours | `+is_resumable = true` (default) |
| Medium | 24 hours | `+WantFlocking = true` (varies) |
| Long | 7 days | Requires special configuration |

If your job exceeds its runtime limit, HTCondor will terminate it. To avoid losing work:

- **Estimate your runtime** from test runs on smaller data.
- **Use checkpointing** for long-running jobs (see below).
- **Break work into smaller chunks** when possible (e.g., train for fewer epochs per job and resume).

See the [CHTC job duration guide](https://chtc.cs.wisc.edu/uw-research-computing/job-duration) for current policies and how to request longer runtimes.


## Checkpointing for long-running jobs

If your training job might exceed the runtime limit, implement **checkpointing** — periodically saving your model state so you can resume from where you left off if the job is interrupted.

For PyTorch, a minimal checkpoint pattern looks like:

```python
# Save checkpoint every N epochs
if epoch % checkpoint_interval == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'checkpoint.pt')
```

```python
# At start of training, resume if checkpoint exists
import os
start_epoch = 0
if os.path.exists('checkpoint.pt'):
    checkpoint = torch.load('checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
```

HTCondor can be configured to automatically transfer checkpoint files when a job is evicted. See the [CHTC checkpointing guide](https://chtc.cs.wisc.edu/uw-research-computing/checkpoint-overview) for details on self-checkpointing and exit-code-based retry workflows.


## Common pitfalls

Here are the mistakes that CHTC facilitators see most often. Avoiding these will save you time and keep the cluster healthy for everyone.

### 1. Running heavy computation on the submit node

The submit node is shared by everyone who logs in. Running training, large data processing, or GPU code directly on the submit node slows it down for all users. **Always submit jobs through HTCondor** — even for quick tests, consider using an interactive job:

```bash
condor_submit -i request_cpus=1 request_memory=4GB
```

### 2. Over-requesting resources

Jobs that request more CPUs, memory, or GPUs than they need wait longer in the queue because fewer machines can satisfy the request. Right-size your requests using the workflow described above.

### 3. Not cleaning up /staging after large jobs

The `/staging` filesystem is shared and has finite capacity. If you leave hundreds of gigabytes of old results sitting there, other researchers may not have space for their active jobs. Clean up after each project or experiment.

### 4. Forgetting to check for held jobs

Held jobs are easy to miss — they sit quietly in the queue. If you submit a batch and walk away without checking, you might come back to find that none of them ran. Always check `condor_q -held` after submitting.

::::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Diagnose the problem

You submitted 50 jobs an hour ago. You run `condor_q` and see:

```
OWNER    BATCH_NAME     SUBMITTED   DONE  RUN  IDLE  HOLD
you      job.sub        3/25 14:00     0    0     0    50
```

All 50 jobs are held. What is your next step, and what command do you use?

::::::::::::::::: solution

Run `condor_q -held` to see the hold reason for your jobs. For example:

```bash
condor_q -held
```

This might show something like:

```
012345.000: Error from slot1@e1234.chtc.wisc.edu: Failed to pull Docker image ...
```

Common hold reasons include:
- Docker image not found or typo in the image name
- Requested more memory or disk than any machine has
- Input file specified in `transfer_input_files` does not exist
- Permissions errors on `/staging` files

Once you fix the underlying issue, release the jobs with `condor_release $USER` or remove them with `condor_rm $USER` and resubmit.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::


## Getting help

CHTC has a dedicated facilitation team that helps researchers optimize their workflows. Do not hesitate to reach out:

- **Email:** [chtc@cs.wisc.edu](mailto:chtc@cs.wisc.edu) — the primary support channel. Include your username, job IDs, and error messages.
- **Office hours:** CHTC holds regular drop-in office hours (check [chtc.cs.wisc.edu](https://chtc.cs.wisc.edu/) for the current schedule). These are great for getting unstuck on tricky job configurations.
- **Documentation:** The [CHTC Guides](https://chtc.cs.wisc.edu/uw-research-computing/) cover everything from getting started to advanced GPU workflows.

:::::::::::::::::::::::::::::::::::::: callout

#### When to email vs. when to debug yourself

Good reasons to email CHTC support:
- Jobs are held with errors you do not understand after checking the documentation
- You need a quota increase for `/staging` or `/home`
- You need access to specific GPU types or longer runtimes
- You are unsure how to structure your workflow for CHTC

Things to try first:
- Check `condor_q -held` and read the hold reason
- Re-read your submit file for typos
- Test with a single short job before submitting a large batch
- Search the [CHTC guides](https://chtc.cs.wisc.edu/uw-research-computing/) for your error message

::::::::::::::::::::::::::::::::::::::::::::::::::


## End-of-session checklist

Before you log off the submit node, run through this checklist:

```bash
# 1. Check for any running jobs
condor_q

# 2. Check for held jobs (fix or remove them)
condor_q -held

# 3. Check home directory usage
du -sh /home/$USER

# 4. Check staging usage
du -sh /staging/$USER

# 5. Remove jobs you no longer need
condor_rm JOB_ID   # or condor_rm $USER to remove all

# 6. Clean up old output files
ls -lh /staging/$USER/
# rm -rf /staging/$USER/old_experiment/   # if no longer needed
```

Make this a habit. Future-you (and your fellow researchers) will thank you.

::::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 3: End-of-session practice

Run through the end-of-session checklist on your submit node right now. Answer these questions:

1. How many jobs (if any) do you have in the queue?
2. Are any of your jobs held? If so, what is the hold reason?
3. How much of your home directory quota are you using?
4. Is there anything in `/staging` that you no longer need?

::::::::::::::::: solution

Run the commands from the checklist above:

```bash
condor_q
condor_q -held
du -sh /home/$USER
quota -vs
du -sh /staging/$USER 2>/dev/null
ls -lh /staging/$USER/ 2>/dev/null
```

Your answers will vary depending on your current session state. The key takeaway is making this check a routine part of your workflow. If you have held jobs, investigate and resolve them. If your storage is getting full, clean up old files before your next session.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::: keypoints

- CHTC is free but shared — right-size your resource requests and clean up after yourself.
- Use `condor_q`, `condor_watch_q`, and `condor_q -held` to monitor jobs; use `condor_rm` to remove jobs you no longer need.
- Check actual resource usage in job logs and ClassAds (`condor_q -af`, `condor_history -af`) to refine future requests.
- Over-requesting CPUs, memory, or GPUs makes your jobs wait longer, not run faster.
- CHTC enforces runtime limits (12 hr, 24 hr, 7 days) — use checkpointing for long-running training jobs.
- Clean up `/home` and `/staging` regularly; run the end-of-session checklist before logging off.
- Email [chtc@cs.wisc.edu](mailto:chtc@cs.wisc.edu) or attend office hours when you need help.

::::::::::::::::::::::::::::::::::::::::::::::::
