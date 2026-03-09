---
title: "CLI Workflows and Automation"
teaching: 15
exercises: 10
---

::::::::::::::::::::::::::::::::::::: questions

- How do I chain multiple jobs into a multi-step workflow on CHTC?
- What is DAGMan and when should I use it?
- How do CHTC CLI commands compare to GCP's `gcloud` CLI?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Write a DAGMan file to chain dependent jobs (data prep → train → evaluate).
- Use advanced HTCondor monitoring commands.
- Compare CHTC and GCP CLI workflows side by side.

::::::::::::::::::::::::::::::::::::::::::::::::

## CLI is the primary interface

In the GCP workshop, [Episode 8](https://qualiamachine.github.io/Intro_GCP_for_ML/) introduces the `gcloud` CLI as an alternative to the web console. On CHTC, **the CLI is the only interface** — everything you've done in this workshop has been via SSH and command-line tools.

This episode covers more advanced patterns: multi-step workflows with DAGMan, monitoring strategies, and automation.

## DAGMan: multi-step workflows

So far, each episode has submitted individual jobs. Real ML workflows often have dependencies:

1. **Data preparation** → produces `.npz` files
2. **Model training** → reads `.npz`, produces `model.pt`
3. **Model evaluation** → reads `model.pt`, produces `evaluation.json`

You could submit these manually one after another, but **DAGMan** (Directed Acyclic Graph Manager) automates the sequencing. Create a `.dag` file:

```
# workflow.dag — data prep → train → evaluate

JOB PREP   prep_data.sub
JOB TRAIN  train_nn_cpu.sub
JOB EVAL   evaluate.sub

PARENT PREP  CHILD TRAIN
PARENT TRAIN CHILD EVAL

RETRY TRAIN 2
```

Submit with:
```bash
condor_submit_dag workflow.dag
```

DAGMan handles:
- Running PREP first, then TRAIN only after PREP succeeds, then EVAL after TRAIN succeeds
- Retrying failed jobs (TRAIN retries up to 2 times)
- Generating a `.dagman.log` file tracking the workflow's progress
- Creating a rescue DAG if something fails partway through

### DAGMan for hyperparameter sweeps

You can also use DAGMan to chain a sweep with post-processing:

```
# sweep_workflow.dag — sweep → aggregate

JOB SWEEP  hpt_sweep.sub
JOB AGG    aggregate.sub

PARENT SWEEP CHILD AGG
```

Here, `SWEEP` submits all HP tuning trials, and `AGG` runs the aggregation script after every trial completes.

## Advanced monitoring

Beyond basic `condor_q`, these commands help with larger workloads:

```bash
# Live-updating status (like watch + condor_q)
condor_watch_q

# Batch view — groups jobs by submit file
condor_q -batch

# Why is my job still idle?
condor_q -better-analyze <cluster_id>

# Check held jobs and their reasons
condor_q -hold

# Release a held job (after fixing the issue)
condor_release <cluster_id>

# Check completed job history
condor_history <cluster_id>

# Show available machines with GPUs
condor_status -compact -constraint 'TotalGpus > 0'

# Check your fair-share priority
condor_userprio -all
```

## CHTC vs. GCP CLI comparison

| Task | GCP (`gcloud`) | CHTC (HTCondor) |
|------|---------------|-----------------|
| Submit a job | `gcloud ai custom-jobs create --config=job.yaml` | `condor_submit job.sub` |
| List running jobs | `gcloud ai custom-jobs list --filter=state=RUNNING` | `condor_q` |
| Check job details | `gcloud ai custom-jobs describe JOB_ID` | `condor_q -l <cluster_id>` |
| Cancel a job | `gcloud ai custom-jobs cancel JOB_ID` | `condor_rm <cluster_id>` |
| List storage | `gcloud storage ls gs://bucket/` | `ls /staging/$USER/` |
| Copy data | `gcloud storage cp local gs://bucket/` | `cp local /staging/$USER/` |
| Check GPU machines | `gcloud compute accelerator-types list` | `condor_status -compact -constraint 'TotalGpus > 0'` |
| Multi-step workflow | Cloud Build YAML or Vertex Pipelines | `condor_submit_dag workflow.dag` |

## Automation patterns

### Shell scripts for repeated experiments

```bash
#!/bin/bash
# run_experiment.sh — Submit a complete experiment pipeline

echo "Preparing data..."
python3 prepare_data.py --input titanic_train.csv

echo "Submitting training sweep..."
condor_submit hpt_sweep.sub

echo "Jobs submitted. Monitor with: condor_watch_q"
echo "After completion, run: python3 aggregate_results.py"
```

### Version control on the submit node

```bash
# Track your experiments with git
cd /home/$USER/workshop/
git init
git add scripts/ submit_files/ params.csv
git commit -m "Initial experiment setup"

# After modifying parameters
git add params.csv
git commit -m "Expand sweep to 50 trials"
```

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Write a DAGMan workflow

1. Create submit files for data prep, training, and evaluation.
2. Write a `workflow.dag` file that chains them.
3. Submit with `condor_submit_dag workflow.dag`.
4. Monitor with `condor_watch_q` and check the `.dagman.log`.

:::::::::::::::: solution

```bash
# Create the DAG file (as shown above)
condor_submit_dag workflow.dag

# Monitor
condor_watch_q

# Check DAGMan progress
tail -f workflow.dag.dagman.log
```

After completion, you should have:
- `train_data.npz` and `val_data.npz` (from PREP)
- `model.pt` and `metrics.json` (from TRAIN)
- `evaluation.json` (from EVAL)

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Command translation

Translate this GCP workflow to CHTC commands:

```bash
# GCP version
gcloud ai custom-jobs create --config=train_config.yaml --region=us-central1
gcloud ai custom-jobs list --filter=state=RUNNING
gcloud ai custom-jobs describe JOB_123
```

:::::::::::::::: solution

```bash
# CHTC equivalent
condor_submit train_nn_gpu.sub
condor_q
condor_q -l <cluster_id>
```

The CHTC commands are shorter and don't require specifying region or authentication — you're already logged into the submit node.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- DAGMan chains dependent jobs automatically: define jobs and parent-child relationships in a `.dag` file.
- Use `condor_watch_q` for live monitoring, `condor_q -better-analyze` for debugging idle jobs.
- CHTC CLI commands are simpler than `gcloud` equivalents — no region, project, or auth flags needed.
- Version control your experiments on the submit node with `git` for reproducibility.

::::::::::::::::::::::::::::::::::::::::::::::::
