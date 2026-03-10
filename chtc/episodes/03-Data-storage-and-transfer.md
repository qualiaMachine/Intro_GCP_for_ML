---
title: "Data Storage and Transfer on CHTC"
teaching: 15
exercises: 10
---

::::::::::::::::::::::::::::::::::::: questions

- How does CHTC's storage hierarchy work, and when should I use each tier?
- How does HTCondor transfer files to and from worker machines?
- How do I get my dataset onto CHTC and make it available to jobs?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Describe CHTC's three-tier storage hierarchy (`/home`, `/scratch`, `/staging`).
- Use HTCondor's `transfer_input_files` and `transfer_output_files` to move data to/from jobs.
- Upload the Titanic dataset and verify a job can read it.

::::::::::::::::::::::::::::::::::::::::::::::::

## CHTC storage hierarchy

CHTC provides three storage tiers, each designed for different use cases:

| Storage | Location | Size limit | Backed up? | Auto-purge? | Best for |
|---------|----------|-----------|------------|-------------|----------|
| **Home** | `/home/<user>/` | 30 GB | Yes | No | Scripts, submit files, small configs |
| **Scratch** | `/scratch/<user>/` | 100 GB | No | Yes (30 days) | Job outputs, intermediate results |
| **Staging** | `/staging/<user>/` | 100 GB | No | No | Large input/output files (>100 MB) |

### Decision tree: where does my data go?

- **Scripts and submit files** (< 1 MB each) → `/home`
- **Training data** (< 100 MB) → `/home` (transferred via `transfer_input_files`)
- **Training data** (100 MB – 4 GB) → `/staging` (use `transfer_input_files` with the full path)
- **Training data** (> 4 GB) → `/staging` with OSDF or Globus transfer (see [CHTC large file guide](https://chtc.cs.wisc.edu/uw-research-computing/file-avail-largedata))
- **Job outputs** → written to `/scratch` or transferred back via HTCondor
- **Model artifacts you want to keep** → move from `/scratch` to `/home` or `/staging`

::::::::::::::::::::::::::::::::::::: callout

### CHTC storage is free

All CHTC storage is **free** — no per-GB charges. The tradeoff: storage has size limits and `/scratch` auto-purges files older than 30 days, so you need to manage your files actively. For comparison, commercial cloud storage typically costs $0.02–$0.10/GB/month.

::::::::::::::::::::::::::::::::::::::::::::::::

## HTCondor file transfer

HTCondor handles moving files between the submit node and worker machines automatically. You control this with submit file directives:

### Input files: submit node → worker

```
transfer_input_files = data.csv, config.json, scripts/preprocess.py
```

- Files are copied to the job's working directory on the worker before execution starts.
- Comma-separated list of files and/or directories.
- Paths are relative to the submit file's directory (or absolute).
- For large files in `/staging`, use the full path: `transfer_input_files = /staging/<user>/large_dataset.csv`

### Output files: worker → submit node

By default, HTCondor transfers **all new and modified files** from the job's working directory back to the submit directory when the job finishes.

To be explicit about what comes back:

```
transfer_output_files = model.pt, metrics.json
```

To rename or redirect output files:

```
transfer_output_remaps = "metrics.json = results/trial_$(Process)/metrics.json"
```

### What happens on the worker

1. HTCondor creates a temporary working directory on the worker.
2. Input files are copied into this directory.
3. Your executable runs with this directory as the current working directory.
4. When the job finishes, output files are transferred back.
5. The temporary directory is cleaned up.

This means **your script can simply read and write local files** — no special file transfer code or cloud storage SDK needed.

## Uploading the Titanic dataset

The Titanic dataset we'll use throughout this workshop is small (~60 KB), so it belongs in `/home`:

```bash
# Clone the workshop repository (if you haven't already)
cd /home/$USER/
git clone https://github.com/qualiaMachine/Intro_GCP_for_ML.git

# The dataset is in the data/ directory
ls -la Intro_GCP_for_ML/data/titanic_train.csv
```

## Testing data transfer with a job

Let's submit a job that reads the Titanic CSV and prints its shape. Create `data_check.py`:

```python
#!/usr/bin/env python3
"""data_check.py — verify that HTCondor can transfer and read a CSV file."""

import pandas as pd
import os

print(f"Working directory: {os.getcwd()}")
print(f"Files available: {os.listdir('.')}")

df = pd.read_csv("titanic_train.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nSurvival rate: {df['Survived'].mean():.2%}")

# Write a small output file to confirm output transfer works
with open("data_summary.txt", "w") as f:
    f.write(f"Shape: {df.shape}\n")
    f.write(f"Columns: {list(df.columns)}\n")
    f.write(f"Survival rate: {df['Survived'].mean():.2%}\n")

print("\nWrote data_summary.txt")
```

Create `data_test.sub`:

```
# data_test.sub — test data transfer with the Titanic dataset

universe     = vanilla
executable   = data_check.py

log          = data_test_$(Cluster).log
output       = data_test_$(Cluster).out
error        = data_test_$(Cluster).err

request_cpus   = 1
request_memory = 1GB
request_disk   = 1GB

# Transfer the script AND the dataset to the worker
transfer_input_files = data_check.py, Intro_GCP_for_ML/data/titanic_train.csv

# Only bring back the summary file (plus default stdout/stderr)
transfer_output_files = data_summary.txt

container_image = docker://continuumio/miniconda3:latest

queue 1
```

Submit and verify:

```bash
chmod +x data_check.py
condor_submit data_test.sub

# After completion:
cat data_test_<cluster_id>.out
cat data_summary.txt
```

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Storage tier quiz

For each of these files, which CHTC storage tier would you use?

1. A 50 KB Python training script
2. A 2 GB imaging dataset used as input to many jobs
3. A 500 MB model checkpoint output from a training job
4. A 15 GB genomics dataset

:::::::::::::::: solution

1. **`/home`** — scripts are small and should be backed up.
2. **`/staging`** — too large for `/home` (30 GB limit shared with other files), and it's reused as input to multiple jobs.
3. **`/scratch`** — intermediate output; move to `/home` or `/staging` if you want to keep it long-term.
4. **`/staging`** with OSDF or Globus transfer — exceeds 4 GB, so standard HTCondor file transfer would be slow.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Transfer the Titanic data

1. Upload the Titanic dataset to your CHTC home directory (clone the repo or copy the file).
2. Create the `data_check.py` and `data_test.sub` files shown above.
3. Submit the job and verify it reads the data correctly.
4. Check that `data_summary.txt` was transferred back.

:::::::::::::::: solution

```bash
# Clone repo (if needed)
cd /home/$USER/
git clone https://github.com/qualiaMachine/Intro_GCP_for_ML.git

# Create scripts in a working directory
mkdir -p workshop && cd workshop
# (create data_check.py and data_test.sub as shown above)

chmod +x data_check.py
condor_submit data_test.sub

# Wait for completion, then check
cat data_test_*.out     # Should show dataset shape (891, 12)
cat data_summary.txt    # Should show the summary
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 3: Storage cost comparison

The Titanic dataset is tiny, but imagine you had a 50 GB dataset. Estimate the monthly storage cost on a commercial cloud platform (~$0.02/GB/month) versus CHTC.

:::::::::::::::: solution

- **Commercial cloud storage:** 50 GB × $0.02/GB/month = **$1.00/month** (plus egress charges if downloading)
- **CHTC `/staging`:** **$0.00/month** (free, within the 100 GB quota)

For this small size the difference is minimal, but for larger datasets (500 GB+) or long-running projects, the savings add up significantly.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- CHTC has three storage tiers: `/home` (30 GB, backed up), `/scratch` (100 GB, auto-purged), and `/staging` (100 GB, for large files).
- HTCondor automatically transfers input files to workers and output files back — your script just reads and writes local files.
- Use `transfer_input_files` to send data to jobs and `transfer_output_files` to control what comes back.
- All CHTC storage is free — no per-GB charges.

::::::::::::::::::::::::::::::::::::::::::::::::
