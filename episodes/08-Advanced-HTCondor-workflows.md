---
title: "Bonus: Advanced HTCondor Workflows"
teaching: 15
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions

- How do I chain multiple HTCondor jobs into a multi-step workflow?
- How can I debug a running job or match jobs to specific hardware?
- What happens when a job fails, and how do I handle retries automatically?

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Write a DAGMan workflow that chains preprocessing, training, and evaluation jobs.
- Submit and monitor DAG workflows with `condor_submit_dag` and `condor_q -dag`.
- Use wrapper scripts, `condor_ssh_to_job`, and ClassAd requirements to control job execution.
- Configure automatic retries and failure handling in DAGMan.
- Access OSPool resources with `+WantFlocking` and `+WantGlidein`.

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: callout

### Bonus episode

This episode is not part of the standard workshop flow. It covers advanced HTCondor features for building multi-step ML workflows on CHTC. Contributions and feedback are welcome — open an issue or pull request on the [lesson repository](https://github.com/qualiaMachine/Intro_GCP_for_ML).

::::::::::::::::::::::::::::::::::::::::::::::::::

## Why multi-step workflows?

In earlier episodes we submitted individual HTCondor jobs — one submit file, one execution. Real ML projects almost always involve a pipeline of steps:

1. **Preprocess** raw data into training-ready format.
2. **Train** a model on the preprocessed data.
3. **Evaluate** the trained model on a held-out test set.

You could submit these one at a time and wait, but that is manual, error-prone, and does not scale. HTCondor's **DAGMan** (Directed Acyclic Graph Manager) lets you define the entire pipeline in a single file and submit it as one unit. DAGMan handles the ordering, monitors each step, and can retry failed jobs automatically.


## DAGMan: Directed Acyclic Graph Manager

A DAG file describes your workflow as a graph of jobs and dependencies. Each node is an HTCondor job (defined by a `.sub` file), and edges define which jobs must finish before others can start.

### DAG file syntax

A DAG file uses a small set of keywords:

- `JOB <name> <submit_file>` — defines a node in the graph.
- `PARENT <name(s)> CHILD <name(s)>` — defines dependency edges.
- `RETRY <name> <count>` — retries a node up to `<count>` times if it fails.

Here is a complete DAG file for a train-then-evaluate pipeline:

```
# workflow.dag
JOB preprocess preprocess.sub
JOB train train.sub
JOB evaluate evaluate.sub

PARENT preprocess CHILD train
PARENT train CHILD evaluate

RETRY train 2
```

This tells DAGMan:

1. Run the `preprocess` job first.
2. When `preprocess` succeeds, run `train`.
3. When `train` succeeds, run `evaluate`.
4. If `train` fails, retry it up to 2 additional times before giving up.

### Submitting a DAG

Submit the entire workflow with a single command:

```bash
condor_submit_dag workflow.dag
```

DAGMan itself runs as a lightweight job on the submit server. It watches the child jobs and advances through the graph as nodes complete.

### Monitoring a DAG

Use `condor_q` with the `-dag` flag to see the DAG structure:

```bash
condor_q -dag
```

This shows each node's status (idle, running, completed, failed) and the overall DAG progress. You can also check the log file that DAGMan creates automatically:

```bash
cat workflow.dag.dagman.out
```

This log records every state transition — when each node was submitted, started, succeeded, or failed.


## Wrapper scripts

Each node in your DAG points to a submit file, and each submit file specifies an `executable`. For ML workflows it is common to use a **wrapper script** — a short shell script that sets up the environment before running your Python code:

```bash
#!/bin/bash
# run_train.sh — wrapper script for the training step

# Unpack the Python environment (transferred as a tarball)
tar -xzf python_env.tar.gz
export PATH=$PWD/python_env/bin:$PATH

# Run the training script
python3 train_nn.py --train data/train.npz --val data/val.npz --epochs 500

# Save exit code so HTCondor sees the real status
exit $?
```

The corresponding submit file references this wrapper:

```
# train.sub
executable = run_train.sh
arguments =

transfer_input_files = python_env.tar.gz, train_nn.py, data/
transfer_output_files = model/, metrics.json

log    = train_$(Cluster).log
output = train_$(Cluster).out
error  = train_$(Cluster).err

request_cpus   = 1
request_memory = 4GB
request_disk   = 2GB

queue
```

::::::::::::::::::::::::::::::::::::: callout

### The `run_*.sh` naming convention

Using a consistent naming pattern like `run_preprocess.sh`, `run_train.sh`, and `run_evaluate.sh` makes it easy to see which wrapper belongs to which pipeline step. Each wrapper handles environment setup so your Python scripts stay portable.

::::::::::::::::::::::::::::::::::::::::::::::::::


## Debugging running jobs with `condor_ssh_to_job`

Sometimes a job is running but producing unexpected output. Instead of waiting for it to fail and reading log files, you can SSH directly into the running job's execution environment:

```bash
condor_ssh_to_job <JOB_ID>
```

This opens a shell session on the execute node, inside the job's working directory. You can inspect files, check environment variables, and even run quick diagnostic commands. When you exit the session, the job continues normally.

```bash
# Example: check what files the job has produced so far
condor_ssh_to_job 12345.0
ls -la
cat metrics.json
exit
```

::::::::::::::::::::::::::::::::::::: callout

### When `condor_ssh_to_job` is not available

Not all pools enable SSH access to running jobs. On CHTC, this feature is generally available for jobs running on CHTC-owned hardware. Jobs running on OSPool resources via flocking may not support it. Check with your system administrators if the command fails.

::::::::::::::::::::::::::::::::::::::::::::::::::


## Job requirements and ClassAds

HTCondor uses a system called **ClassAds** (Classified Advertisements) to match jobs to machines. Every machine advertises its properties (CPUs, memory, GPU type, operating system), and every job advertises its requirements. The HTCondor matchmaker pairs them up.

### Requesting specific hardware

You can add a `requirements` line to your submit file to target specific hardware:

```
# Request a machine with at least 8 CPUs and a GPU
request_cpus   = 8
request_memory = 16GB
request_gpus   = 1

# Only run on machines with NVIDIA A100 GPUs
requirements = (CUDADeviceName == "NVIDIA A100-SXM4-80GB")
```

### Viewing available ClassAds

To see what machines are available and what they advertise:

```bash
# List all GPUs available in the pool
condor_status -compact -constraint 'TotalGpus > 0'

# See detailed ClassAds for a specific machine
condor_status -long <machine_name>
```


## Job priorities and scheduling

When the pool is busy, HTCondor decides which jobs run first based on **priority**. You can influence this with the `priority` keyword in your submit file:

```
# Higher numbers = higher priority (runs sooner)
priority = 10
```

Within your own jobs, higher-priority jobs will be scheduled before lower-priority ones. Note that this only affects your relative ordering — it does not let you jump ahead of other users. HTCondor uses a **fair-share** scheduling policy across users.

You can check your current priority standing with:

```bash
condor_userprio
```


## Handling job failures

Jobs fail for many reasons: out-of-memory errors, network timeouts, transient hardware issues. HTCondor and DAGMan provide several mechanisms for handling failures gracefully.

### Retries in DAGMan

The simplest approach is the `RETRY` keyword in your DAG file:

```
RETRY train 2
```

If the `train` node exits with a non-zero exit code, DAGMan will resubmit it up to 2 more times. This is useful for transient failures (e.g., a preempted job or a temporary network error).

### Holding and releasing jobs

In your submit file, you can use **policy expressions** to hold a job that exits abnormally and release it after a delay:

```
# Hold the job if it exits with a non-zero code
on_exit_hold = (ExitCode != 0)
on_exit_hold_reason = "Job exited with non-zero code; holding for inspection."

# Automatically release held jobs after 5 minutes (300 seconds),
# but only if fewer than 3 release attempts have been made
periodic_release = (HoldReasonCode == 3) && (NumJobStarts < 3) && \
                   ((time() - EnteredCurrentStatus) > 300)
```

This pattern is helpful when failures are intermittent — the job is held so you can inspect it, but it also gets a second chance automatically.

### Checking why a job was held

```bash
condor_q -hold <JOB_ID>
```

This shows the hold reason, which tells you whether the job ran out of memory, hit a time limit, or failed for another reason.


## Accessing OSPool resources with flocking

CHTC is part of the **OSPool** (Open Science Pool), a nationwide network of computing resources. By adding two lines to your submit file, your jobs can "flock" to machines at other institutions when CHTC is busy:

```
+WantFlocking = true
+WantGlidein = true
```

- `+WantFlocking` allows your jobs to run on OSPool resources contributed by other institutions.
- `+WantGlidein` allows your jobs to run on resources provisioned dynamically by GlideinWMS.

::::::::::::::::::::::::::::::::::::: callout

### Flocking considerations

When your jobs flock to remote sites, there are a few things to keep in mind:

- **Transfer everything** — remote machines do not have access to your home directory. Make sure all input files are listed in `transfer_input_files`.
- **Software portability** — pack your Python environment into a tarball or use a container. Do not rely on software installed on CHTC submit servers.
- **Longer queue times** — flocked jobs may wait longer to start since they compete with other OSPool users.
- **No `condor_ssh_to_job`** — you typically cannot SSH into jobs running on remote sites.

::::::::::::::::::::::::::::::::::::::::::::::::::


## Putting it all together: a complete DAG workflow

Here is a full example of a three-step ML pipeline managed by DAGMan.

**Step 1: Write the submit files**

```
# preprocess.sub
executable = run_preprocess.sh
transfer_input_files = python_env.tar.gz, preprocess.py, raw_data/
transfer_output_files = processed_data/

log    = preprocess_$(Cluster).log
output = preprocess_$(Cluster).out
error  = preprocess_$(Cluster).err

request_cpus   = 1
request_memory = 4GB
request_disk   = 2GB

queue
```

```
# train.sub
executable = run_train.sh
transfer_input_files = python_env.tar.gz, train_nn.py, processed_data/
transfer_output_files = model/, metrics.json

log    = train_$(Cluster).log
output = train_$(Cluster).out
error  = train_$(Cluster).err

request_cpus   = 4
request_memory = 8GB
request_disk   = 4GB
request_gpus   = 1

+WantFlocking = true
+WantGlidein = true

queue
```

```
# evaluate.sub
executable = run_evaluate.sh
transfer_input_files = python_env.tar.gz, evaluate.py, model/, processed_data/
transfer_output_files = results/

log    = evaluate_$(Cluster).log
output = evaluate_$(Cluster).out
error  = evaluate_$(Cluster).err

request_cpus   = 1
request_memory = 4GB
request_disk   = 2GB

queue
```

**Step 2: Write the DAG file**

```
# ml_pipeline.dag
JOB preprocess preprocess.sub
JOB train train.sub
JOB evaluate evaluate.sub

PARENT preprocess CHILD train
PARENT train CHILD evaluate

RETRY train 2
RETRY evaluate 1
```

**Step 3: Submit and monitor**

```bash
# Submit the full pipeline
condor_submit_dag ml_pipeline.dag

# Watch the DAG progress
condor_q -dag

# Check the DAGMan log for detailed status
tail -f ml_pipeline.dag.dagman.out
```

When the DAG completes successfully, all three steps have run in sequence, and your results are in the `results/` directory.


::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1 — Write a DAG file

Given two submit files `clean_data.sub` and `train_model.sub`, write a DAG file that:

1. Runs `clean_data` first.
2. Runs `train_model` after `clean_data` succeeds.
3. Retries `train_model` up to 3 times on failure.

:::::::::::::::::::::::::::::::::::: solution

```
# my_pipeline.dag
JOB clean_data clean_data.sub
JOB train_model train_model.sub

PARENT clean_data CHILD train_model

RETRY train_model 3
```

Submit with:

```bash
condor_submit_dag my_pipeline.dag
```

:::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2 — Add GPU requirements

You have a training job that needs an NVIDIA GPU with at least 40 GB of memory. Modify the submit file snippet below to add the appropriate requirements:

```
executable = run_train.sh
request_cpus = 4
request_memory = 16GB
request_gpus = 1

# Add your requirements line here

queue
```

:::::::::::::::::::::::::::::::::::: solution

```
executable = run_train.sh
request_cpus = 4
request_memory = 16GB
request_gpus = 1

requirements = (CUDAGlobalMemoryMb >= 40000)

+WantFlocking = true
+WantGlidein = true

queue
```

The `CUDAGlobalMemoryMb` ClassAd attribute reports GPU memory in megabytes. Adding flocking increases the chance of matching a machine with a large GPU. You can discover available GPU ClassAd attributes by running `condor_status -compact -constraint 'TotalGpus > 0'`.

:::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 3 — Diagnose a held job

You check on your jobs and see one is held:

```
$ condor_q

-- Schedd: submit1.chtc.wisc.edu
 ID      OWNER    SUBMITTED     RUN_TIME  ST PRI SIZE CMD
 98765.0 jdoe     3/25 10:30   0+00:00:00 H  0   4.0 run_train.sh
```

What command would you run to find out why it is held? What are two common reasons a job might be placed on hold?

:::::::::::::::::::::::::::::::::::: solution

Run:

```bash
condor_q -hold 98765.0
```

Two common reasons:

1. **Exceeded memory request** — the job used more memory than `request_memory` and was killed by the system.
2. **`on_exit_hold` policy** — the job exited with a non-zero exit code and the submit file included `on_exit_hold = (ExitCode != 0)`.

Other possibilities include exceeding disk quota, missing input files, or a Docker image that could not be pulled. The hold reason message from `condor_q -hold` will tell you exactly which one occurred.

:::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: keypoints

- DAGMan lets you define multi-step workflows in a single `.dag` file using `JOB` and `PARENT/CHILD` syntax.
- Submit an entire pipeline with `condor_submit_dag` and monitor it with `condor_q -dag`.
- Wrapper scripts (`run_*.sh`) set up the execution environment before calling your Python code.
- `condor_ssh_to_job` lets you debug a running job by opening a shell on the execute node.
- ClassAd `requirements` expressions let you match jobs to specific hardware (e.g., GPU type or memory).
- DAGMan `RETRY`, `on_exit_hold`, and `periodic_release` provide automatic failure handling.
- `+WantFlocking` and `+WantGlidein` give your jobs access to the nationwide OSPool when CHTC is busy.

::::::::::::::::::::::::::::::::::::::::::::::::::
