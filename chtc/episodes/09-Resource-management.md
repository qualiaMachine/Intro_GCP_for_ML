---
title: "Resource Management on CHTC"
teaching: 15
exercises: 10
---

::::::::::::::::::::::::::::::::::::: questions

- How does CHTC's fair-share scheduling work?
- What are common job failures and how do I diagnose them?
- How do I handle job eviction and runtime limits with checkpointing?
- How should I clean up after completing my work?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Check your fair-share priority with `condor_userprio`.
- Diagnose and fix common job failures (held jobs, memory exceeded, wall-time exceeded).
- Implement self-checkpointing for long-running training jobs.
- Clean up old files and practice good resource citizenship.

::::::::::::::::::::::::::::::::::::::::::::::::

## CHTC is free — but resource management still matters

Unlike cloud platforms where bad resource management costs you money, CHTC's costs are social: over-requesting resources takes capacity from other users, and running too many jobs lowers your fair-share priority.

### No billing risk, but good citizenship matters

| What can go wrong | Impact |
|----------|----------------|
| Over-requesting CPUs/memory/GPUs | Resources sit idle, unavailable to other users |
| Running too many jobs at once | Your fair-share priority drops, future jobs wait longer |
| Not cleaning up old files | `/home` and `/staging` fill up, blocking new work |
| Under-requesting resources | Jobs get held or evicted mid-run |

## Fair-share priority

CHTC uses **fair-share scheduling** to distribute resources equitably. The more jobs you run recently, the lower your priority becomes temporarily — giving other users a chance.

```bash
# Check your current priority
condor_userprio

# Check all users' priorities
condor_userprio -all
```

A **lower number = higher priority**. New users start with priority 0.5. Running many jobs increases your number (lowers priority), but it recovers over time as you use fewer resources.

Priority recovers over time as you use fewer resources — it's a temporary adjustment, not a permanent penalty.

## Diagnosing common job failures

### Held jobs

Jobs move to "held" state when something goes wrong. Check held jobs:

```bash
# Show held jobs with reasons
condor_q -hold

# Show detailed hold reason for a specific job
condor_q -l <cluster_id> | grep HoldReason
```

Common hold reasons and fixes:

| Hold Reason | Cause | Fix |
|-------------|-------|-----|
| `Memory usage exceeded request_memory` | Job used more RAM than requested | Increase `request_memory` |
| `Disk usage exceeded request_disk` | Job wrote more data than allocated | Increase `request_disk` |
| `Job was evicted` | Worker needed for higher-priority work | Re-submit (normal behavior) |
| `Cannot find container image` | Docker Hub image doesn't exist | Check image name/tag |
| `SHADOW/STARTER failure` | Worker machine issue | Release and retry: `condor_release <id>` |

Release a held job (after fixing the issue):
```bash
condor_release <cluster_id>
```

### Memory exceeded

The most common failure. If your job is held for memory:

1. Check actual memory usage in the `.log` file
2. Set `request_memory` to ~1.5x the actual usage
3. Don't request 100 GB "just in case" — this wastes pool resources

```bash
# Check actual memory usage from a completed job
grep "Memory (MB)" <job>.log
```

### Wall-time limits and eviction

All CHTC jobs have a **72-hour default runtime limit**. GPU jobs have additional constraints:

| GPU tier | Time limit | Submit file directive |
|----------|-----------|----------------------|
| GPU Lab (shared) | **4 hours** (interactive) | Interactive sessions only |
| CHTC-owned GPUs | **72 hours** | (default) |
| Research group backfill GPUs | **No guarantee** | `+is_resumable = true` |
| Longer GPU slots | **24 hours** | `+WantMediumGpuJobs = true` |
| Longest GPU slots | **7 days** | `+WantLongGpuJobs = true` |

**Backfill GPUs** are GPUs owned by specific research groups. Your job can run on them when the group isn't using them, but it can be **preempted at any time** when the owning group reclaims the resource. To opt in:

```
+is_resumable = true
```

This gets you access to more GPUs but with no runtime guarantees — making checkpointing essential.

Even on CHTC-owned hardware, jobs can be evicted if the machine needs maintenance or a higher-priority user needs resources. **Any job that runs for more than a few hours should implement checkpointing.**

## Checkpointing for long-running jobs

Checkpointing lets your training job survive eviction and runtime limits. The idea: periodically save your full training state (model weights, optimizer state, epoch number, best metrics), exit with a special code, and HTCondor restarts your job from where it left off.

### How HTCondor self-checkpointing works

```
┌───────────────┐     exit 85      ┌───────────────┐     exit 85
│  Start fresh  │ ──────────────── │  Resume ep 50 │ ──────────────
│  ep 1 → 49    │  transfer ckpt   │  ep 50 → 99   │  transfer ckpt
└───────────────┘                  └───────────────┘
                                                          ...
┌───────────────┐     exit 0
│  Resume ep 150│ ──────────────── Done!
│  ep 150 → 200 │  transfer outputs
└───────────────┘
```

1. Your script saves `checkpoint.pt` with full training state
2. It exits with code **85** (the checkpoint exit code)
3. HTCondor transfers `checkpoint.pt` back to the submit node
4. HTCondor immediately restarts your job (on the same or different machine)
5. Your script detects `checkpoint.pt` and resumes training
6. When training finishes (early stop or max epochs), it exits with code **0**

### The training script

Our `train_nn.py` already supports checkpointing with two flags:

```bash
python3 train_nn.py \
    --train train_data.npz \
    --val val_data.npz \
    --epochs 10000 \
    --checkpoint_every 3600 \
    "$@"
```

- `--checkpoint_every 3600` — save checkpoint and exit every 3600 seconds (1 hour)
- `--checkpoint_exit_code 85` — exit code that tells HTCondor "restart me" (default 85)

The checkpoint saves: model weights, optimizer state, epoch counter, validation history, early stopping state, and best weights. On restart, training resumes exactly where it left off.

::::::::::::::::::::::::::::::::::::: callout

### Checkpoint frequency

CHTC recommends checkpointing every **1–5 hours**, with a maximum of 10 hours between checkpoints. Checkpointing too frequently (under 1 hour) means more time is spent on I/O than training. Too infrequently means more lost work if evicted.

::::::::::::::::::::::::::::::::::::::::::::::::

### The submit file

The key submit file directives for checkpointing:

```
# train_nn_checkpoint.sub — Training with self-checkpointing

universe     = vanilla
executable   = run_nn_checkpoint.sh

request_cpus   = 1
request_memory = 2GB
request_disk   = 2GB
request_gpus   = 1

transfer_input_files = train_nn.py, run_nn_checkpoint.sh, train_data.npz, val_data.npz
transfer_output_files = model.pt, metrics.json, eval_history.csv, checkpoint.pt

# Exit code 85 = "checkpoint taken, please restart me"
checkpoint_exit_code = 85

# Only transfer final outputs on normal exit (code 0)
when_to_transfer_output = ON_EXIT

# Also save checkpoint files if evicted mid-run
transfer_checkpoint_files = checkpoint.pt

# Opt in to backfill GPUs (more GPUs available, but can be preempted)
+is_resumable = true

container_image = docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

arguments = --epochs 10000 --learning_rate 0.0005 --patience 50

queue 1
```

**Key directives explained:**

| Directive | Purpose |
|-----------|---------|
| `checkpoint_exit_code = 85` | Tells HTCondor that exit code 85 means "checkpoint taken, restart job" |
| `when_to_transfer_output = ON_EXIT` | Only transfer final outputs on normal completion (exit 0) |
| `transfer_checkpoint_files = checkpoint.pt` | Also save checkpoint if job is evicted (not just on voluntary exit) |
| `+is_resumable = true` | Opt in to backfill GPUs — more capacity, but preemption possible |

### What happens at each stage

1. **First run:** No `checkpoint.pt` exists → training starts from epoch 1. After 1 hour, saves checkpoint, exits 85.
2. **Restart:** HTCondor transfers `checkpoint.pt` to new machine → script resumes from saved epoch. Trains for another hour, saves checkpoint, exits 85.
3. **Repeat** until training finishes (early stop or max epochs).
4. **Final run:** Training completes → script removes `checkpoint.pt`, saves final `model.pt` and `metrics.json`, exits 0.

::::::::::::::::::::::::::::::::::::: callout

### Wrapper-based timeout fallback

If your training script doesn't have built-in checkpoint timing (unlike our `train_nn.py`), you can use a shell wrapper with `timeout`:

```bash
#!/bin/bash
# Timeout after 4 hours, leaving time for checkpoint transfer
timeout 4h python3 train.py "$@"
status=$?
if [ $status -eq 124 ]; then
    # timeout killed the process — checkpoint should exist from periodic saves
    exit 85
fi
exit $status
```

The `timeout` command sends SIGTERM after the specified duration (4h). Your Python script should catch SIGTERM and save a checkpoint before exiting. This approach works for any training script, not just ours.

::::::::::::::::::::::::::::::::::::::::::::::::

### When you need checkpointing

For the Titanic dataset in this workshop, training completes in seconds — checkpointing isn't needed. But for your real research with:

- Large datasets (ImageNet, genomics data)
- Deep models (ResNets, transformers) requiring hours/days of training
- GPU jobs that may be evicted

...checkpointing is essential. The pattern shown here scales directly to larger models — just save/restore your model and optimizer state.

## Right-sizing resource requests

**Don't over-request.** Requesting more than you need doesn't make your job faster — it just means HTCondor has fewer machines that can run it, leading to longer queue times for you and less capacity for others.

| Resource | Start with | Increase if |
|----------|-----------|-------------|
| `request_cpus` | 1 | Training is CPU-parallelized |
| `request_memory` | 2 GB | Job held for memory; check `.log` for actual usage |
| `request_disk` | 2 GB | Job held for disk; large model/data files |
| `request_gpus` | 1 | Need multi-GPU (rare for single-model training) |

## Storage cleanup

```bash
# Check your storage usage
du -sh /home/$USER/
du -sh /scratch/$USER/
du -sh /staging/$USER/

# Clean up old job outputs
find /scratch/$USER/ -name "*.log" -mtime +7 -delete
find /scratch/$USER/ -name "*.out" -mtime +7 -delete
find /scratch/$USER/ -name "*.err" -mtime +7 -delete

# Or use the workshop cleanup script
bash cleanup.sh              # Dry run
bash cleanup.sh --execute    # Actually delete
```

Remember:
- `/scratch` auto-purges files older than 30 days
- `/home` is backed up but limited to 30 GB — don't store large outputs here
- `/staging` has no auto-purge — clean up manually when done with large files

## Cleanup checklist

After completing this workshop:

- [ ] Remove jobs from the queue: `condor_rm $USER`
- [ ] Clean up trial directories from HP sweeps
- [ ] Remove temporary model files you don't need
- [ ] Move any files you want to keep from `/scratch` to `/home` or download
- [ ] Check `/staging` for large files you no longer need

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Check your priority

1. Run `condor_userprio` to see your current priority.
2. If you ran HP sweep jobs, has your priority changed compared to the start of the workshop?
3. How long would it take for your priority to recover to baseline?

:::::::::::::::: solution

```bash
condor_userprio
```

Your priority number reflects recent resource usage. After running the workshop exercises, it may have increased slightly. Priority decays exponentially — it typically returns to baseline within 1–2 days of reduced usage.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Diagnose a held job

A job is held with the message: `Memory usage 2048 MB exceeded request_memory of 1024 MB`. What should you do?

:::::::::::::::: solution

1. The job used 2 GB of RAM but only 1 GB was requested.
2. Update your submit file: `request_memory = 3GB` (1.5x the actual usage).
3. Release and retry, or remove and re-submit:

```bash
# Option 1: Fix submit file, release the held job
condor_release <cluster_id>

# Option 2: Remove and re-submit with corrected request
condor_rm <cluster_id>
condor_submit corrected_job.sub
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 3: Workshop cleanup

1. Run the cleanup script in dry-run mode to see what would be deleted.
2. Review the list and run with `--execute` to clean up.
3. Check your storage usage with `du -sh`.

:::::::::::::::: solution

```bash
cd /home/$USER/workshop/
bash cleanup.sh                # See what would be deleted
bash cleanup.sh --execute      # Actually delete
du -sh /home/$USER/            # Check remaining usage
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 4: Add checkpointing to a training job

Look at `train_nn_checkpoint.sub`. What would happen if:
1. The job is evicted at epoch 75 (before the 1-hour checkpoint)?
2. The job checkpoints at epoch 100, but the restart machine has no GPU?

:::::::::::::::: solution

1. **Eviction at epoch 75:** Because we set `transfer_checkpoint_files = checkpoint.pt`, HTCondor will try to transfer the last saved checkpoint. If the eviction is graceful, the checkpoint from the most recent voluntary save is used. If the machine loses power, training restarts from the beginning (no checkpoint was transferred). This is why checkpoint frequency matters — you lose at most one checkpoint interval of work.

2. **Restart on CPU:** The script uses `torch.load(..., map_location=device)` which handles this automatically. If the checkpoint was saved on GPU, it will be loaded onto CPU. Training will be slower but will resume correctly. HTCondor tries to match `request_gpus`, but if you want to guarantee GPU, use `require_gpus` constraints.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- CHTC is free, but good resource citizenship matters: don't over-request, clean up when done.
- Fair-share priority means heavy recent usage temporarily lowers your queue priority.
- Most held jobs are caused by exceeding memory or disk limits — check `.log` files for actual usage.
- Use `checkpoint_exit_code = 85` and `transfer_checkpoint_files` for long-running jobs that may exceed time limits or be evicted.
- GPU jobs have time limits (12h default) — request longer slots with `+WantMediumGpuJobs` or implement checkpointing.
- Clean up old outputs regularly, especially in `/scratch` and `/staging`.

::::::::::::::::::::::::::::::::::::::::::::::::
