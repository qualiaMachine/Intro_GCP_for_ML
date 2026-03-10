---
title: "Resource Management on CHTC"
teaching: 10
exercises: 5
---

::::::::::::::::::::::::::::::::::::: questions

- How does CHTC's fair-share scheduling work?
- What are common job failures and how do I diagnose them?
- How should I clean up after completing my work?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Check your fair-share priority with `condor_userprio`.
- Diagnose and fix common job failures (held jobs, memory exceeded, wall-time exceeded).
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

### Wall-time exceeded

CHTC GPU jobs have time limits. If your training job exceeds the limit:

- **12 hours (short):** default for GPU jobs
- **24 hours (medium):** request with `+WantMediumGpuJobs = true`
- **7 days (long):** request with `+WantLongGpuJobs = true`

Add to your submit file:
```
+WantMediumGpuJobs = true
```

For very long training, implement **checkpointing** — save model state periodically and resume from the latest checkpoint.

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

::::::::::::::::::::::::::::::::::::: keypoints

- CHTC is free, but good resource citizenship matters: don't over-request, clean up when done.
- Fair-share priority means heavy recent usage temporarily lowers your queue priority.
- Most held jobs are caused by exceeding memory or disk limits — check `.log` files for actual usage.
- Clean up old outputs regularly, especially in `/scratch` and `/staging`.

::::::::::::::::::::::::::::::::::::::::::::::::
