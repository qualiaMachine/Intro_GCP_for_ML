---
title: "Connecting to CHTC"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions

- How do I connect to CHTC and start working on a submit node?
- What can (and can't) I do on the submit node?
- What tools do I need to know to check the status of HTCondor and my jobs?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Log in to a CHTC submit node via SSH.
- Navigate the CHTC filesystem and understand the purpose of the `/home` directory.
- Distinguish between the submit node (controller) and execute nodes (workers).
- Run basic HTCondor commands to inspect the pool and job queue.
- Set up a working environment by cloning the workshop repository and downloading data.

::::::::::::::::::::::::::::::::::::::::::::::::

## Connecting to a submit node

All work in this workshop begins on a **submit node** — a shared server that acts as your control plane for launching HTCondor jobs. Think of it the same way you would think of a lightweight controller notebook in a cloud workflow: you use it to prepare code, stage data, and submit jobs, but you never run heavy computation on it directly.

CHTC provides several submit nodes. For this workshop, connect to one of the access points (your instructor will confirm which one to use):

```bash
ssh username@ap2002.chtc.wisc.edu
```

Replace `username` with your UW-Madison NetID. You will authenticate with your UW-Madison password (and possibly Duo two-factor authentication, depending on the server configuration).

::::::::::::::::::::::::::::::::::::: callout

#### First time connecting?

The first time you SSH into a new server, you'll see a message like:

```
The authenticity of host 'ap2002.chtc.wisc.edu' can't be established.
ED25519 key fingerprint is SHA256:...
Are you sure you want to continue connecting (yes/no)?
```

Type `yes` and press Enter. This adds the server's fingerprint to your `~/.ssh/known_hosts` file so you won't be asked again.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: callout

#### Connecting from different operating systems

- **macOS / Linux:** Open a terminal and use the `ssh` command shown above.
- **Windows:** Use the built-in **Windows Terminal** or **PowerShell** (both include an SSH client on Windows 10+). Alternatively, install [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/) or [MobaXterm](https://mobaxterm.mobatek.net/).
- **Chromebook / tablet:** Use a browser-based SSH client or install an SSH app from your platform's store.

::::::::::::::::::::::::::::::::::::::::::::::::

## Navigating the CHTC filesystem

Once logged in, you land in your **home directory** (`/home/username`). This is your primary workspace on the submit node.

```bash
pwd
# /home/username

ls -la
```

### Home directory basics

Your `/home` directory has a quota of approximately **20 GB**. It is designed for:

- Submit files (`.sub`) and job scripts
- Small code repositories
- Configuration files and logs

It is **not** designed for large datasets. For bigger files, CHTC provides `/staging` and **SQUID** — we'll cover those in [Episode 3](03-Data-storage-and-access.md).

Check your current disk usage and quota:

```bash
quota -vs
```

::::::::::::::::::::::::::::::::::::: callout

#### CHTC storage tiers at a glance

| Location | Purpose | Typical quota | Persists between jobs? |
|----------|---------|---------------|----------------------|
| `/home` | Code, submit files, small inputs/outputs | ~20 GB | Yes |
| `/staging` | Large datasets, model checkpoints | ~200 GB+ (by request) | Yes |
| **SQUID** (`/squid`) | Large read-only data shared across many jobs | By request | Yes |
| Job working directory | Temporary scratch space on execute node | Varies | No — cleaned up after job completes |

::::::::::::::::::::::::::::::::::::::::::::::::

## Submit node vs. execute node

This distinction is central to everything we do in this workshop:

| | Submit node | Execute node |
|--|-------------|-------------|
| **What it is** | The shared server you SSH into | A worker machine assigned by HTCondor |
| **Who uses it** | Many researchers, simultaneously | One job (or a few), temporarily |
| **What to do here** | Edit code, write submit files, submit jobs, check job status | Run your actual computation (training, inference, preprocessing) |
| **What NOT to do** | Run training loops, load large models, GPU-intensive work | Nothing — HTCondor manages these automatically |

The submit node is a **shared resource**. Running heavy computation on it slows things down for every other user on the same server. HTCondor enforces this by design: you describe what you need in a submit file, and HTCondor finds an execute node with matching resources (CPUs, memory, GPUs) to run your job.

This is the same "controller" pattern used in cloud workflows — a lightweight orchestrator delegates expensive work to dedicated compute. The difference is that on CHTC, the "controller" is a shared server rather than a personal VM, and the "compute" is managed by HTCondor rather than a cloud API.

## Basic HTCondor commands

Before we submit any jobs, let's get familiar with the two most important HTCondor commands for checking the state of the system.

### `condor_status` — What resources are available?

```bash
condor_status
```

This shows all the execute nodes (also called "slots") in the HTCondor pool. You'll see columns for the machine name, operating system, architecture, state (e.g., `Unclaimed`, `Claimed`), and resource details.

To see a summary instead of the full list:

```bash
condor_status -total
```

To check what GPU resources are available:

```bash
condor_status -constraint 'TotalGPUs > 0' -compact
```

### `condor_q` — What jobs are in the queue?

```bash
condor_q
```

This shows **your** jobs in the queue. Right now it should be empty — we haven't submitted anything yet. You'll see output like:

```
-- Schedd: ap2002.chtc.wisc.edu : <...>
OWNER    BATCH_NAME    SUBMITTED   DONE  RUN  IDLE  TOTAL  JOB_IDS

Total for query: 0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended
```

To see jobs from all users (useful for understanding how busy the system is):

```bash
condor_q -all
```

::::::::::::::::::::::::::::::::::::: callout

#### Other useful HTCondor commands

You'll use these in later episodes, but here's a preview:

| Command | Purpose |
|---------|---------|
| `condor_submit job.sub` | Submit a job described in `job.sub` |
| `condor_q -hold` | Show held jobs and the reason they're held |
| `condor_rm <job_id>` | Remove (cancel) a job |
| `condor_history` | Show your completed jobs |
| `condor_status -gpus` | Show GPU availability across the pool |

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Explore the HTCondor pool

Run `condor_status -total` and answer the following:

1. How many total slots (machines) are in the pool?
2. How many are currently `Unclaimed` (idle) vs. `Claimed` (running a job)?
3. Based on what you see, is the cluster busy or relatively free right now?

:::::::::::::::: solution

The output of `condor_status -total` shows a summary table with rows for each machine state. Look for:

- **Total**: the total number of slots available.
- **Unclaimed**: slots available to run new jobs.
- **Claimed**: slots currently running someone's job.

If most slots are `Unclaimed`, the cluster has plenty of capacity. If most are `Claimed`, your jobs may wait in the queue before starting. This is normal on a shared system — HTCondor manages the queue fairly.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Setting up your working environment

Now let's prepare the files we'll use for the rest of the workshop. We'll clone the workshop repository and verify that everything is in place.

### Clone the workshop repository

```bash
cd ~
git clone https://github.com/qualiaMachine/Intro_GCP_for_ML.git
cd Intro_GCP_for_ML
ls
```

This repository contains:

- **Submit files** (`.sub`) for each episode's HTCondor jobs.
- **Training scripts** (Python) that will run on execute nodes.
- **Sample data** and configuration files.

### Download the workshop dataset

Some episodes require a small dataset that isn't stored in the Git repository. Download it into your working directory:

```bash
cd ~/Intro_GCP_for_ML
wget -q https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/data/penguins.csv -P data/
ls -lh data/
```

::::::::::::::::::::::::::::::::::::: callout

#### Keep your home directory tidy

Your `/home` quota is limited. A few good habits:

- **Don't store large datasets in `/home`.** Use `/staging` for anything over a few hundred MB (covered in [Episode 3](03-Data-storage-and-access.md)).
- **Clean up job output files** (`*.log`, `*.out`, `*.err`) after you've reviewed them.
- **Remove old Conda/pip caches** if you install packages locally. These can grow quickly.
- **Check usage regularly** with `quota -vs`.

::::::::::::::::::::::::::::::::::::::::::::::::

### Verify your setup

Run a quick check to make sure everything is ready:

```bash
echo "Home directory: $HOME"
echo "Current directory: $(pwd)"
echo "Files in repo:"
ls ~/Intro_GCP_for_ML/
echo ""
echo "HTCondor status:"
condor_q
```

You should see the cloned repository contents and an empty job queue. If `condor_q` returns an error, double-check that you're on a CHTC submit node (not a different server).

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Submit node etiquette

A colleague tells you they ran a deep learning training script directly on the submit node because "it was faster than writing a submit file." What problems could this cause, and what should they do instead?

:::::::::::::::: solution

Running heavy computation on the submit node causes several problems:

- **It slows down the server for everyone.** The submit node is shared by many researchers. A single training job can consume most of the CPU and memory, making the server sluggish for others trying to edit files, submit jobs, or check job status.
- **It won't have GPU access.** Submit nodes typically don't have GPUs attached, so the training would run on CPU only — much slower than using a GPU execute node.
- **It's not reproducible.** Running interactively means there's no submit file to re-run later, no automatic logging, and no record of what resources were used.
- **CHTC may kill the process.** Administrators monitor submit nodes and may terminate long-running or resource-heavy processes without warning.

**What to do instead:** Write a submit file (`.sub`) that describes the job's resource needs and let HTCondor run it on an appropriate execute node. We'll do exactly this starting in [Episode 4](04-Training-models-in-VertexAI.md).

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## What's next

You're now connected to CHTC, familiar with the filesystem, and have the workshop materials ready. In the next episode, we'll set up data storage and learn how to move data between the submit node, `/staging`, and your HTCondor jobs.

::::::::::::::::::::::::::::::::::::: keypoints

- Connect to CHTC via SSH to a submit node (e.g., `ap2002.chtc.wisc.edu`) — this is your controller for the workshop.
- The submit node is for lightweight work only: editing code, writing submit files, and managing jobs. Heavy computation goes to execute nodes via HTCondor.
- Use `condor_status` to check available resources and `condor_q` to check your job queue.
- Your `/home` directory (~20 GB) holds code and submit files; larger data belongs in `/staging` or SQUID.
- Clone the workshop repository to get submit files and training scripts for the remaining episodes.

::::::::::::::::::::::::::::::::::::::::::::::::
