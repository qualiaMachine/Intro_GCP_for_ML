---
title: CHTC Resources Reference
---

## Storage Hierarchy

| Storage | Path | Quota | Backed up | Auto-purge | Use for |
|---------|------|-------|-----------|------------|---------|
| Home | `/home/<user>/` | 30 GB | Yes | No | Scripts, configs, small files |
| Scratch | `/scratch/<user>/` | 100 GB | No | 30 days | Job outputs, intermediate data |
| Staging | `/staging/<user>/` | 100 GB | No | No | Large input/output files |

## GPU Lab Hardware

| GPU | VRAM | CUDA Capability | Typical availability |
|-----|------|----------------|---------------------|
| A100 40 GB | 40 GB | 8.0 | Good |
| A100 80 GB | 80 GB | 8.0 | Moderate |
| H100 | 80 GB | 9.0 | Limited |
| H200 | 141 GB | 9.0 | Limited |

Request specific GPUs with `require_gpus` in your submit file:
```
# At least 40 GB GPU memory
require_gpus = (GlobalMemoryMb >= 40000)

# CUDA capability 9.0+
require_gpus = (Capability >= 9.0)
```

## Job Time Limits

| Category | Max runtime | How to request |
|----------|-------------|---------------|
| Short (default) | 12 hours | Default for GPU jobs |
| Medium | 24 hours | `+WantMediumGpuJobs = true` |
| Long | 7 days | `+WantLongGpuJobs = true` |

## HTCondor Command Reference

| Command | Purpose |
|---------|---------|
| `condor_submit job.sub` | Submit a job |
| `condor_submit_dag workflow.dag` | Submit a DAGMan workflow |
| `condor_submit -i job.sub` | Start an interactive job |
| `condor_q` | Check your jobs |
| `condor_q -batch` | Batch view of your jobs |
| `condor_q -hold` | Show held jobs with reasons |
| `condor_q -better-analyze <id>` | Diagnose why a job is idle |
| `condor_q -l <id>` | Show all job attributes |
| `condor_watch_q` | Live-updating job status |
| `condor_status -compact` | Show available machines |
| `condor_status -constraint 'TotalGpus > 0'` | Show GPU machines |
| `condor_rm <id>` | Remove (cancel) a job |
| `condor_rm $USER` | Remove all your jobs |
| `condor_release <id>` | Release a held job |
| `condor_history <id>` | Check completed job details |
| `condor_userprio` | Check your fair-share priority |

## Submit File Quick Reference

```
# Minimal submit file template
universe     = vanilla
executable   = my_script.py

log          = job_$(Cluster).log
output       = job_$(Cluster).out
error        = job_$(Cluster).err

request_cpus   = 1
request_memory = 2GB
request_disk   = 2GB

transfer_input_files = my_script.py, data.csv
transfer_output_files = results.json

container_image = docker://python:3.11-slim

arguments = --flag value

queue 1
```

### GPU job additions
```
request_gpus = 1
require_gpus = (GlobalMemoryMb >= 40000)
+WantMediumGpuJobs = true
```

### Sweep syntax
```
arguments = --lr $(lr) --patience $(p)
queue lr, p from params.csv
```

## Useful Links

- [CHTC Homepage](https://chtc.cs.wisc.edu/)
- [CHTC GPU Lab Guide](https://chtc.cs.wisc.edu/uw-research-computing/gpu-lab)
- [HTCondor Submit File Reference](https://htcondor.readthedocs.io/en/latest/users-manual/submitting-a-job.html)
- [CHTC File Transfer Guide](https://chtc.cs.wisc.edu/uw-research-computing/file-availability)
- [Apptainer/Docker on CHTC](https://chtc.cs.wisc.edu/uw-research-computing/docker-jobs)
- [DAGMan Documentation](https://htcondor.readthedocs.io/en/latest/users-manual/dagman-workflows.html)
- [CHTC Office Hours](https://chtc.cs.wisc.edu/uw-research-computing/get-help) — drop-in help sessions
