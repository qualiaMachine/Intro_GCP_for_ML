---
title: "The Submit Node and Your First HTCondor Job"
teaching: 20
exercises: 10
---

::::::::::::::::::::::::::::::::::::: questions

- How do I connect to CHTC and navigate the filesystem?
- What is an HTCondor submit file and how do I write one?
- How do I submit, monitor, and inspect jobs?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Connect to a CHTC submit node via SSH and explore the storage hierarchy.
- Write an HTCondor submit file to run a simple Python script.
- Use `condor_submit`, `condor_q`, and other HTCondor commands to manage jobs.
- Understand the role of containers on CHTC.

::::::::::::::::::::::::::::::::::::::::::::::::

## Connecting to the submit node

The **submit node** is your home base on CHTC. It's a shared Linux machine where you write code, prepare data, write submit files, and launch jobs. Think of it as a lightweight control center — **do not run heavy computation directly on the submit node**, as it's shared with other users.

Connect via SSH:

```bash
ssh <username>@submit1.chtc.wisc.edu
```

Replace `<username>` with your CHTC username. You'll land in your home directory (`/home/<username>/`).

::::::::::::::::::::::::::::::::::::: callout

### Tips for SSH sessions

- Use **tmux** or **screen** to keep sessions alive if your connection drops: `tmux new -s workshop`
- To reconnect to an existing tmux session: `tmux attach -t workshop`
- Consider setting up SSH keys for passwordless login (see the [CHTC SSH guide](https://chtc.cs.wisc.edu/uw-research-computing/connecting))

::::::::::::::::::::::::::::::::::::::::::::::::

## Exploring the filesystem

Once logged in, take a look at the storage hierarchy introduced in Episode 1:

```bash
# Your home directory (30 GB, backed up)
ls -la /home/$USER/

# Your scratch space (100 GB, NOT backed up, auto-purged after 30 days)
ls -la /scratch/$USER/

# Your staging area (100 GB, for large file transfers)
ls -la /staging/$USER/
```

For this workshop, we'll keep scripts and submit files in `/home` (they're small) and use `/scratch` for job outputs.

## Your first submit file

An HTCondor **submit file** (`.sub`) tells HTCondor everything it needs to run your job: what to execute, what files to transfer, and what resources to request. Here's a minimal example:

Create a file called `hello.sub`:

```
# hello.sub — a minimal HTCondor submit file

universe     = vanilla
executable   = hello.py

# Log, output, and error files
log          = hello_$(Cluster).log
output       = hello_$(Cluster).out
error        = hello_$(Cluster).err

# Resource requests
request_cpus   = 1
request_memory = 512MB
request_disk   = 512MB

# Transfer the Python script to the worker
transfer_input_files = hello.py

# Use a container with Python installed
container_image = docker://python:3.11-slim

queue 1
```

Key elements:
- **`executable`**: The program to run. For Python scripts, this is the script itself (with a `#!/usr/bin/env python3` shebang) or a wrapper shell script.
- **`log`/`output`/`error`**: Where HTCondor writes job metadata, stdout, and stderr. `$(Cluster)` is replaced with your job's cluster ID.
- **`request_cpus`/`request_memory`/`request_disk`**: Resources your job needs. HTCondor matches your job to a machine with these resources available.
- **`transfer_input_files`**: Files to copy from the submit node to the worker. Comma-separated list.
- **`container_image`**: The Docker/Apptainer image to run inside. CHTC pulls this image automatically.
- **`queue 1`**: Submit 1 copy of this job.

## The hello.py script

Create a simple test script:

```python
#!/usr/bin/env python3
"""hello.py — verify that HTCondor can run a Python script."""

import sys
import os

print(f"Hello from HTCondor!")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Files in working directory: {os.listdir('.')}")
print(f"Hostname: {os.uname().nodename}")
```

Make it executable:

```bash
chmod +x hello.py
```

## Submitting and monitoring

Submit the job:

```bash
condor_submit hello.sub
```

You'll see output like:

```
Submitting job(s).
1 job(s) submitted to cluster 12345678.
```

Monitor your job:

```bash
# Check status of your jobs
condor_q

# More detail about a specific job
condor_q -better-analyze <cluster_id>

# Watch jobs in real-time (refreshes every 2 seconds)
condor_watch_q

# Check all available machines
condor_status -compact
```

Job states you'll see in `condor_q`:
- **I** (Idle) — waiting for a matching machine
- **R** (Running) — executing on a worker
- **H** (Held) — something went wrong; needs investigation
- **C** (Completed) — finished (only shown with `-all` flag)

## Inspecting results

Once your job completes (disappears from `condor_q`), check the output files:

```bash
# Standard output from your script
cat hello_<cluster_id>.out

# Any errors
cat hello_<cluster_id>.err

# HTCondor's own log (timestamps, resource usage, exit code)
cat hello_<cluster_id>.log
```

The `.log` file is especially useful — it shows when the job started, which machine it ran on, how much memory/disk it actually used, and the exit code.

## Containers on CHTC

CHTC worker machines have a base operating system but **not** the Python packages your ML code needs. Containers solve this by packaging your entire software environment:

- **Docker images** are specified with `container_image = docker://image:tag` in your submit file
- HTCondor pulls the image automatically and runs your job inside it
- Common images: `docker://python:3.11-slim`, `docker://continuumio/miniconda3`, `docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`

For custom environments, you can build an **Apptainer definition file** (`.def`) and create a `.sif` image. We'll do this in later episodes.

::::::::::::::::::::::::::::::::::::: callout

### Container images are cached

The first time a container image is used on a worker, it must be pulled (downloaded). This can take a few minutes for large images. Subsequent jobs on the same worker reuse the cached image. For large images (PyTorch GPU ~7 GB), consider pre-building a `.sif` file and transferring it as an input file for faster startup.

::::::::::::::::::::::::::::::::::::::::::::::::

## Interactive jobs

Sometimes you need to debug interactively on a worker machine. HTCondor supports interactive jobs:

```bash
condor_submit -i hello.sub
```

This gives you a shell session on a worker machine with your requested resources. Interactive jobs have a **4-hour time limit** on CHTC. Use them for:
- Debugging scripts that fail as batch jobs
- Testing that your container has the right packages
- Quick exploratory analysis on GPU hardware

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: What belongs on the submit node?

Which of these tasks should you do on the submit node, and which should be submitted as HTCondor jobs?

1. Editing a Python script
2. Training an XGBoost model on a 10 GB dataset
3. Running `condor_q` to check job status
4. A hyperparameter sweep with 50 trials
5. Inspecting a 100-line CSV output file
6. Generating embeddings for 10,000 documents

:::::::::::::::: solution

**Submit node:** 1, 3, 5 — lightweight editing, job management, and inspection.

**HTCondor jobs:** 2, 4, 6 — compute-heavy tasks that need dedicated hardware. The submit node is shared with other users and shouldn't be used for heavy processing.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Submit and inspect

1. Create the `hello.py` script and `hello.sub` submit file shown above.
2. Submit the job with `condor_submit hello.sub`.
3. Monitor it with `condor_q` until it completes.
4. Read the `.out`, `.err`, and `.log` files. What machine did your job run on? How much memory did it actually use?

:::::::::::::::: solution

After the job completes:

```bash
cat hello_<cluster_id>.out    # Should show "Hello from HTCondor!" and system info
cat hello_<cluster_id>.err    # Should be empty if no errors
cat hello_<cluster_id>.log    # Shows machine name, memory/disk usage, exit code
```

In the `.log` file, look for lines like:
- `Job executing on host: <IP_ADDRESS>` — the worker machine
- `Memory (MB): Usage` — actual memory used vs. requested
- `Normal termination (return value 0)` — clean exit

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Essential HTCondor commands

| Command | Purpose |
|---------|---------|
| `condor_submit job.sub` | Submit a job |
| `condor_q` | Check your job queue |
| `condor_q -hold` | Show held jobs with hold reasons |
| `condor_watch_q` | Live-updating job status |
| `condor_status -compact` | Show available machines |
| `condor_rm <cluster_id>` | Remove (cancel) a job |
| `condor_rm <username>` | Remove all your jobs |
| `condor_history <cluster_id>` | Check completed job details |
| `condor_q -better-analyze <cluster_id>` | Debug why a job is idle |

::::::::::::::::::::::::::::::::::::: keypoints

- The submit node is your control center — use it for lightweight tasks and job management, not heavy compute.
- HTCondor submit files specify what to run, what files to transfer, and what resources to request.
- Use `condor_submit` to launch jobs, `condor_q` to monitor them, and inspect `.out`/`.err`/`.log` files for results.
- Containers (`container_image`) provide reproducible software environments on CHTC workers.
- Interactive jobs (`condor_submit -i`) give you a shell on a worker for debugging (4-hour limit).

::::::::::::::::::::::::::::::::::::::::::::::::
