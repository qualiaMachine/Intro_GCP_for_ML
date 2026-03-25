---
title: "Training Models on CHTC with HTCondor"
teaching: 25
exercises: 15
---

:::::::::::::::::::::::::::::::::::::: questions

- How do I write an HTCondor submit file to run a training job on CHTC?
- How do I use Docker containers to manage my software environment on CHTC?
- How do I monitor my jobs and troubleshoot failures?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Write an HTCondor submit file that runs an XGBoost training script inside a Docker container.
- Test a training script locally on the submit node before submitting to the cluster.
- Submit, monitor, and inspect the results of an HTCondor training job.
- Interpret HTCondor job states (Idle, Running, Held, Completed) and use log files for debugging.
- Request appropriate compute resources (CPUs, memory, disk) for a training job.

::::::::::::::::::::::::::::::::::::::::::::::::

In [Episode 1](01-Introduction.md) we introduced the "submit node as controller" pattern — your submit node is a lightweight machine where you write code, prepare data, and launch jobs. The actual training happens on powerful execute nodes managed by HTCondor. In this episode we put that pattern into practice by submitting an XGBoost training job to CHTC.

## The training script: train_xgboost.py

We will use the same `train_xgboost.py` script introduced earlier in this workshop. It trains an XGBoost classifier on the Titanic dataset, accepts hyperparameters via command-line arguments, and saves a serialized model artifact. The key design feature is that the script is **self-contained** — it reads a local CSV file, trains the model, and writes output files to the current working directory. This makes it ideal for HTCondor, where each job runs in its own isolated scratch directory on an execute node.

The script accepts the following arguments:

- `--train` — path to the training CSV file
- `--max_depth` — maximum tree depth (controls model complexity)
- `--eta` — learning rate
- `--subsample` — fraction of rows sampled per boosting round
- `--colsample_bytree` — fraction of features sampled per tree
- `--num_round` — number of boosting iterations

::::::::::::::::::::::::::::::::::::::: challenge

### Understanding the training script

Review `scripts/train_xgboost.py` and answer the following:

1. What preprocessing steps does the script apply before training?
2. What output file(s) does the script produce?
3. How would you change the number of boosting rounds from the command line?

::::::::::::::::::::::::::::::::::::::: solution

### Solution

1. The script fills missing values (`Age` with median, `Embarked` with mode), maps categorical fields to numeric values (`Sex` to 0/1, `Embarked` to 0/1/2), and drops non-predictive columns (`PassengerId`, `Name`, `Ticket`, `Cabin`).
2. It produces a file called `xgboost-model` — a serialized XGBoost Booster object saved with `joblib`.
3. Pass `--num_round 200` (or any integer) on the command line.

:::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::

## Testing locally on the submit node

Before submitting to the cluster, always test your script on the submit node with a quick run. This catches bugs, missing dependencies, and data issues before you wait in the job queue.

::::::::::::::::::::::::::::::::::::: callout

### Keep local tests small

The submit node is a shared resource — do not run heavy computations on it. For local testing, use a small dataset or a small number of training rounds (e.g., `--num_round 5`). The goal is to verify that the script runs without errors, not to produce a good model.

:::::::::::::::::::::::::::::::::::::

```bash
$ python3 train_xgboost.py \
    --train titanic_train.csv \
    --max_depth 3 \
    --eta 0.1 \
    --num_round 5
```

If the script finishes without errors and produces an `xgboost-model` file, you are ready to submit to HTCondor. Remove the test output before submitting:

```bash
$ rm -f xgboost-model
```

## Writing an HTCondor submit file

An HTCondor submit file (`.sub`) tells HTCondor everything it needs to run your job: what to execute, which files to transfer, what resources to request, and where to write logs. Here is a complete submit file for our XGBoost training job:

```
# train_xgboost.sub — HTCondor submit file for XGBoost training

universe = docker
docker_image = python:3.10

executable = train_xgboost.py
arguments = --train titanic_train.csv --max_depth 3 --eta 0.1 --num_round 100

transfer_input_files = train_xgboost.py, titanic_train.csv
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

log    = job_$(Cluster).log
output = job_$(Cluster).out
error  = job_$(Cluster).err

request_cpus   = 1
request_memory = 2GB
request_disk   = 1GB

queue 1
```

Let's walk through each section.

### Container environment

```
universe = docker
docker_image = python:3.10
```

These two lines tell HTCondor to run your job inside a Docker container. The `python:3.10` image from Docker Hub provides a clean Python environment. If your script needs additional packages (like `xgboost`, `pandas`, `scikit-learn`), you have two options:

1. **Add a `pip install` step** — make your executable a wrapper shell script that installs packages before running the Python script.
2. **Build a custom Docker image** — create an image with all dependencies pre-installed and push it to Docker Hub. This is the recommended approach for reproducibility and faster job startup.

::::::::::::::::::::::::::::::::::::: callout

### Using a custom Docker image

For production workflows, build a Docker image that includes all your dependencies. For example, if you have a `Dockerfile`:

```dockerfile
FROM python:3.10
RUN pip install xgboost==2.1.0 pandas scikit-learn joblib
```

Build and push it to Docker Hub:

```bash
$ docker build -t yourusername/xgboost-train:v1 .
$ docker push yourusername/xgboost-train:v1
```

Then reference it in your submit file:

```
universe = docker
docker_image = yourusername/xgboost-train:v1
```

This avoids installing packages every time a job runs, which saves time and ensures consistent environments.

:::::::::::::::::::::::::::::::::::::

### Executable and arguments

```
executable = train_xgboost.py
arguments = --train titanic_train.csv --max_depth 3 --eta 0.1 --num_round 100
```

The `executable` is the script HTCondor will run. The `arguments` line passes command-line arguments, just as you would on the command line. Note that the file paths in `arguments` are relative to the job's working directory on the execute node — HTCondor creates a temporary scratch directory for each job and places transferred files there.

### File transfer

```
transfer_input_files = train_xgboost.py, titanic_train.csv
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
```

HTCondor copies the listed input files from your submit directory to the execute node before the job starts. When the job finishes (`ON_EXIT`), any new files created in the working directory are transferred back to the submit directory. This is how you get your trained model (`xgboost-model`) back.

### Log, output, and error files

```
log    = job_$(Cluster).log
output = job_$(Cluster).out
error  = job_$(Cluster).err
```

HTCondor writes three files for each job:

| File | Contents |
|------|----------|
| `.log` | HTCondor system events: job submitted, started, finished, evicted, resource usage |
| `.out` | Everything your script writes to **stdout** (`print()` statements) |
| `.err` | Everything your script writes to **stderr** (warnings, errors, tracebacks) |

The `$(Cluster)` macro is replaced with the job's cluster ID (a unique number), so each submission produces uniquely named files.

### Resource requests

```
request_cpus   = 1
request_memory = 2GB
request_disk   = 1GB
```

These tell HTCondor what your job needs. HTCondor uses these to match your job to a machine with sufficient resources.

::::::::::::::::::::::::::::::::::::: callout

### How to choose resource requests

- **Start small.** Request only what you need. Over-requesting wastes shared resources and can make your job wait longer in the queue.
- **Check actual usage.** After a job completes, look at the `.log` file for a "Partitionable Resources" summary that shows how much memory and disk the job actually used. Use this to refine future requests.
- **Common starting points for ML jobs:**
  - Small tabular data (XGBoost, sklearn): 1 CPU, 2 GB memory, 1 GB disk
  - Medium neural networks: 1 CPU + 1 GPU, 8 GB memory, 5 GB disk
  - Large models (fine-tuning LLMs): 1 CPU + 1 GPU (A100/H100), 32–64 GB memory, 20+ GB disk

:::::::::::::::::::::::::::::::::::::

### The queue command

```
queue 1
```

This tells HTCondor to submit one instance of the job. In [Episode 6](06-Hyperparameter-tuning.md), we will use `queue` with multiple arguments to submit many jobs at once for hyperparameter sweeps.

## Submitting the job

Once your submit file is ready, submit it with `condor_submit`:

```bash
$ condor_submit train_xgboost.sub
```

You should see output like:

```
Submitting job(s).
1 job(s) submitted to cluster 1234567.
```

The **cluster ID** (here `1234567`) is the unique identifier for your submission. You will use it to monitor the job and find your output files.

## Monitoring your job

### condor_q — check the job queue

```bash
$ condor_q
```

This shows all your jobs currently in the queue. A typical output looks like:

```
OWNER    BATCH_NAME       SUBMITTED   DONE  RUN  IDLE  TOTAL  JOB_IDS
user     job_1234567     3/25 14:02     -     1     -     1   1234567.0
```

### condor_watch_q — live updates

```bash
$ condor_watch_q
```

This provides a live-updating view of your jobs, similar to `watch condor_q`. Press `Ctrl+C` to exit.

### Understanding job states

| State | Meaning |
|-------|---------|
| **Idle (I)** | Job is waiting for a matching execute node. This is normal — it may take seconds to minutes depending on cluster load and your resource requests. |
| **Running (R)** | Job is actively executing on an execute node. |
| **Held (H)** | Something went wrong and HTCondor has paused the job. Check the hold reason with `condor_q -hold`. Common causes include requesting more resources than available, Docker image pull failures, or file transfer errors. |
| **Completed (C)** | Job finished. Check `.out`, `.err`, and `.log` files for results. |

::::::::::::::::::::::::::::::::::::: callout

### What to do when a job is held

A held job will not run until you fix the problem. To see why a job is held:

```bash
$ condor_q -hold
```

Common hold reasons and fixes:

- **"Docker image not found"** — Check for typos in `docker_image`. Verify the image exists on Docker Hub.
- **"Failed to transfer input files"** — Make sure all files listed in `transfer_input_files` exist in your submit directory.
- **"Memory limit exceeded"** — Your job used more memory than requested. Increase `request_memory` and resubmit.

After fixing the issue, you can release the held job:

```bash
$ condor_release <cluster_id>
```

Or remove it and resubmit:

```bash
$ condor_rm <cluster_id>
```

:::::::::::::::::::::::::::::::::::::

## Checking results after completion

Once your job disappears from `condor_q` (meaning it has completed), check the output files in your submit directory:

```bash
$ ls job_1234567.*
job_1234567.log  job_1234567.out  job_1234567.err
```

### Inspect stdout

```bash
$ cat job_1234567.out
```

This should show the output from your training script, including dataset sizes, training time, and where the model was saved.

### Check for errors

```bash
$ cat job_1234567.err
```

If the job succeeded, this file is typically empty or contains only minor warnings. If the job failed, the Python traceback will appear here.

### Check the HTCondor log

```bash
$ cat job_1234567.log
```

The log file contains system-level information about your job's lifecycle. At the end of a completed job, you will see a resource usage summary:

```
Partitionable Resources :    Usage  Request Allocated
   Cpus                 :        1        1         1
   Disk (KB)            :   150000  1048576   4110820
   Memory (MB)          :      450     2048      2048
```

This tells you how much of each resource your job actually used compared to what you requested. Use this to right-size future requests — if you requested 2 GB of memory but only used 450 MB, you can safely reduce `request_memory` to `1GB` next time.

### Check for the model artifact

```bash
$ ls -lh xgboost-model
```

If this file exists, your training job completed successfully and transferred the model back to the submit node.

::::::::::::::::::::::::::::::::::::::: challenge

### Submit and monitor a training job

1. Create a file called `train_xgboost.sub` with the submit file contents shown above.
2. Make sure `train_xgboost.py` and `titanic_train.csv` are in the same directory.
3. Submit the job with `condor_submit train_xgboost.sub`.
4. Monitor it with `condor_q` and `condor_watch_q`.
5. After the job completes, examine the `.out`, `.err`, and `.log` files.
6. Verify that the `xgboost-model` file was transferred back.

**Bonus:** Look at the resource usage summary in the `.log` file. How much memory did the job actually use? Could you reduce `request_memory` for future runs?

::::::::::::::::::::::::::::::::::::::: solution

### Solution

```bash
$ condor_submit train_xgboost.sub
Submitting job(s).
1 job(s) submitted to cluster 1234567.

$ condor_q
OWNER    BATCH_NAME       SUBMITTED   DONE  RUN  IDLE  TOTAL  JOB_IDS
user     job_1234567     3/25 14:02     -     1     -     1   1234567.0

$ cat job_1234567.out
# (training output: dataset size, training time, model saved message)

$ cat job_1234567.err
# (should be empty or contain minor warnings)

$ ls -lh xgboost-model
-rw-r--r-- 1 user user 48K Mar 25 14:05 xgboost-model
```

For the bonus: check the `Partitionable Resources` section of the `.log` file. The Titanic dataset is small, so memory usage will likely be well under 1 GB. You could safely reduce `request_memory` to `1GB` for this job, though 2 GB provides a comfortable margin.

:::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::: challenge

### Diagnose a held job

Suppose you submit a job and see this in `condor_q`:

```
OWNER    BATCH_NAME       SUBMITTED   DONE  RUN  IDLE  HOLD  TOTAL  JOB_IDS
user     job_1234568     3/25 14:10     -     -     -     1     1   1234568.0
```

The job is held. What steps would you take to diagnose and fix the problem?

::::::::::::::::::::::::::::::::::::::: solution

### Solution

1. Run `condor_q -hold` to see the hold reason. For example:
   ```
   1234568.0: Error from slot: Failed to pull Docker image 'python:3.1'
   ```
2. The hold reason tells you the Docker image name is wrong — `python:3.1` does not exist (the correct tag is `python:3.10`).
3. Fix the `docker_image` line in your submit file.
4. Remove the held job with `condor_rm 1234568`.
5. Resubmit with `condor_submit train_xgboost.sub`.

The general debugging workflow is: **check the hold reason** with `condor_q -hold`, **fix the underlying issue** in your submit file or script, **remove the broken job**, and **resubmit**.

:::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::

## Summary of useful HTCondor commands

| Command | Purpose |
|---------|---------|
| `condor_submit file.sub` | Submit a job |
| `condor_q` | Check your jobs in the queue |
| `condor_watch_q` | Live-updating job status |
| `condor_q -hold` | See why a job is held |
| `condor_release <id>` | Release a held job |
| `condor_rm <id>` | Remove a job from the queue |
| `condor_history <id>` | Check details of a completed job |

::::::::::::::::::::::::::::::::::::: keypoints

- **Test locally first**: Always run your training script on the submit node with a small test before submitting to HTCondor.
- **Submit files are declarative**: An HTCondor `.sub` file specifies the executable, container image, input files, resource requests, and log file locations — everything HTCondor needs to run your job.
- **Docker containers provide reproducibility**: Use `universe = docker` and `docker_image` to run jobs in a consistent software environment across different execute nodes.
- **File transfer is automatic**: HTCondor transfers input files to the execute node before the job starts and transfers output files back when it finishes.
- **Monitor and debug with HTCondor tools**: Use `condor_q`, `condor_watch_q`, and `condor_q -hold` to track job status, and inspect `.out`, `.err`, and `.log` files to diagnose problems.
- **Right-size your resource requests**: Check actual resource usage in the `.log` file and adjust `request_cpus`, `request_memory`, and `request_disk` accordingly.

::::::::::::::::::::::::::::::::::::::::::::::::
