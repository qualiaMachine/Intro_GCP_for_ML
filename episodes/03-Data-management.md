---
title: "Data Management on CHTC"
teaching: 30
exercises: 15
---

:::::::::::::::::::::::::::::::::::::: questions

- What storage options are available on CHTC, and which should I use for different types of data?
- How does HTCondor transfer files to and from jobs?
- How do I prepare datasets like the Titanic CSV files for use in HTCondor jobs?
- What are CHTC's data policies, quotas, and best practices?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Identify the three main CHTC storage tiers (`/home`, `/staging`, and SQUID) and explain when to use each one.
- Use HTCondor's `transfer_input_files` and `transfer_output_files` directives to move data into and out of jobs.
- Check disk usage and manage files within CHTC quota limits.
- Prepare the Titanic dataset for training jobs submitted through HTCondor.
- Apply best practices for file sizes, storage etiquette, and sensitive data on shared research infrastructure.

::::::::::::::::::::::::::::::::::::::::::::::::

ML/AI projects depend on data, so understanding where to store files and how to get them into your jobs is essential. Unlike cloud platforms such as GCP or AWS, CHTC does not charge for storage — but shared resources come with **quotas** and **community etiquette** expectations that are just as important to follow.

> #### Consult your institution before handling sensitive data on CHTC
> **Do not store restricted or sensitive data (HIPAA, FERPA, proprietary) on CHTC systems unless explicitly approved by your institution's IT or compliance office.** CHTC general-use systems are not certified for regulated data. If you work with sensitive datasets, contact the [CHTC facilitation team](https://chtc.cs.wisc.edu/uw-research-computing/get-help) to discuss options.

## CHTC storage tiers overview

CHTC provides three main storage locations. Each is designed for a different purpose and has different size limits.

| Location | Typical Quota | Purpose | Accessible from jobs? |
|-----------|--------------|---------|----------------------|
| `/home/<netid>` | ~20 GB | Code, scripts, submit files, small config files | Yes (default working directory) |
| `/staging/<netid>` | ~100+ GB (by request) | Larger datasets transferred to/from jobs | Yes (via `transfer_input_files`) |
| SQUID (`/squid/<netid>`) | ~20 GB (by request) | Large, **read-only** data shared across many jobs via HTTP | Yes (via HTTP URL in `transfer_input_files`) |

::::::::::::::::::::::::::::::::::::: callout

#### No storage charges, but quotas matter

Unlike GCS buckets where you pay per GB per month, CHTC storage is free. However, these are **shared filesystems** — exceeding your quota or storing unnecessary files affects every researcher on the cluster. Treat quotas the way you would treat a shared lab refrigerator: take only the space you need, label your things, and clean up when you are done.

::::::::::::::::::::::::::::::::::::::::::::::::

### `/home` — your home directory

Your home directory at `/home/<netid>` is where you land when you log in to a CHTC submit server. It is suitable for:

- Submit files (`.sub`)
- Executable scripts (`.sh`, `.py`)
- Small configuration and parameter files
- Small output logs

The default quota is approximately **20 GB**. Do **not** store large datasets here — it is backed by a shared network filesystem and is not designed for high-throughput I/O.

### `/staging` — for larger data

The `/staging/<netid>` directory is intended for datasets that are too large for `/home` but need to be transferred into jobs. Typical use cases:

- Training datasets (CSV, Parquet, image archives)
- Pre-trained model weights
- Large output files produced by jobs

You must **request** a `/staging` directory by emailing [chtc@cs.wisc.edu](mailto:chtc@cs.wisc.edu) or filling out a quota request. Default allocations start around 100 GB and can be increased for justified research needs.

Files in `/staging` are referenced in your submit file using `transfer_input_files` and are copied to the job's working directory at runtime.

### SQUID — large read-only data via HTTP

SQUID (located at `/squid/<netid>` on the submit server) is a web-cache-based system for distributing **large, read-only** files to many jobs simultaneously. Data placed in SQUID is served via HTTP, so it is ideal for:

- Reference datasets shared across hundreds or thousands of jobs
- Pre-trained model files that every job needs but never modifies

To use SQUID data in a job, reference it by its HTTP URL:

```
transfer_input_files = http://proxy.chtc.wisc.edu/SQUID/<netid>/my_large_file.tar.gz
```

SQUID is **not** for output or frequently changing files. For complete details, see the [CHTC large data guide](https://chtc.cs.wisc.edu/uw-research-computing/file-avail-largedata).

::::::::::::::::::::::::::::::::::::: callout

#### Which storage tier should I use?

A quick rule of thumb based on file size:

- **< 100 MB per file** and < 500 MB total per job: `/home` is fine.
- **100 MB – a few GB per file**: use `/staging` with `transfer_input_files`.
- **> 1 GB read-only data shared across many jobs**: consider SQUID.
- **Very large data (> 10 GB per file)**: contact the CHTC facilitation team for guidance. You may need to split the data or use special transfer mechanisms.

::::::::::::::::::::::::::::::::::::::::::::::::

## Checking your disk usage

Before adding new files, check how much space you are using. Log in to your CHTC submit server and run:

```bash
# Check home directory usage
du -sh /home/<netid>
```

```bash
# Check quota (if your system supports it)
quota -s
```

```bash
# Check staging usage (if you have a staging directory)
du -sh /staging/<netid>
```

To find the largest files and directories:

```bash
# Top 10 largest items in your home directory
du -h /home/<netid> | sort -rh | head -10
```

::::::::::::::::::::::::::::::::::::: callout

#### Keep an eye on your usage

Run `du -sh` regularly, especially after jobs complete. Output files can accumulate quickly if you run many jobs. Remove or move results you no longer need on the submit server.

::::::::::::::::::::::::::::::::::::::::::::::::

## How HTCondor file transfer works

HTCondor jobs run on **execute machines** that are separate from the submit server. Your job does not have direct access to `/home` or `/staging` on the submit server while it runs. Instead, HTCondor copies files back and forth using its **file transfer mechanism**.

### Sending files to a job: `transfer_input_files`

In your submit file, list the files your job needs:

```
executable = train_model.sh
arguments  = titanic_train.csv

transfer_input_files = /staging/<netid>/titanic_train.csv, train_model.py

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

log    = job_$(Cluster).log
output = job_$(Cluster).out
error  = job_$(Cluster).err

request_cpus   = 1
request_memory = 2GB
request_disk   = 1GB

queue
```

Key points:

- `transfer_input_files` accepts a comma-separated list of file paths (absolute or relative to the submit directory) and URLs.
- All listed files are placed in the job's **top-level working directory** on the execute machine, regardless of their original directory structure.
- The `executable` script is transferred automatically — you do not need to list it again.

### Getting files back: `transfer_output_files`

By default, HTCondor transfers **all new and modified files** from the job's working directory back to the submit directory when the job finishes. If you only want specific outputs returned, use:

```
transfer_output_files = model_output.pkl, metrics.csv
```

This is good practice — it avoids transferring temporary files or large intermediate results you do not need.

::::::::::::::::::::::::::::::::::::: callout

#### Avoid transferring unnecessary files

If your job creates large temporary files (e.g., extracted archives, intermediate checkpoints), either delete them in your script before the job exits or use `transfer_output_files` to specify only what you need. Transferring unnecessary data wastes network bandwidth and fills up your submit directory.

::::::::::::::::::::::::::::::::::::::::::::::::

## Preparing the Titanic dataset for HTCondor jobs

The Titanic dataset is small (under 100 KB total for both CSVs), so it fits comfortably in `/home`. For this workshop, we will keep it simple and transfer the files directly from the submit directory.

### 1. Download the data

On the CHTC submit server:

```bash
# Navigate to your home directory
cd /home/<netid>

# Create a working directory for this workshop
mkdir -p ml-workshop && cd ml-workshop

# Download the dataset
wget https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/data/data.zip

# Unzip
unzip data.zip
```

Verify the files are present:

```bash
ls -lh *.csv
```

You should see `titanic_train.csv` and `titanic_test.csv`, each well under 1 MB.

### 2. Reference the data in a submit file

Create a minimal submit file to verify that file transfer works:

```
# test_transfer.sub
executable = test_transfer.sh

transfer_input_files = titanic_train.csv, titanic_test.csv

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

log    = test_$(Cluster).log
output = test_$(Cluster).out
error  = test_$(Cluster).err

request_cpus   = 1
request_memory = 512MB
request_disk   = 100MB

queue
```

Create the corresponding script:

```bash
#!/bin/bash
# test_transfer.sh — verify that input files arrived
echo "=== Files in working directory ==="
ls -lh

echo ""
echo "=== First 5 lines of titanic_train.csv ==="
head -5 titanic_train.csv

echo ""
echo "=== Row counts ==="
echo "Train rows: $(wc -l < titanic_train.csv)"
echo "Test rows:  $(wc -l < titanic_test.csv)"
```

Make the script executable and submit:

```bash
chmod +x test_transfer.sh
condor_submit test_transfer.sub
```

Check the status of your job:

```bash
condor_q
```

Once the job completes, inspect the output:

```bash
cat test_*.out
```

You should see the file listing, the first few rows of the training data, and the row counts, confirming that HTCondor successfully transferred both CSV files into the job.

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Transfer files from `/staging`

Suppose you have a larger dataset (e.g., 500 MB of image files packaged as `images.tar.gz`) stored in `/staging/<netid>/`. Write a submit file snippet that transfers this archive into a job and extracts it. What would the executable script look like?

:::::::::::::::: solution

**Submit file snippet:**

```
transfer_input_files = /staging/<netid>/images.tar.gz, process_images.py
```

**Executable script (`process_images.sh`):**

```bash
#!/bin/bash
# Extract the archive
tar -xzf images.tar.gz

# Run processing
python3 process_images.py

# Clean up extracted files before job exits to avoid transferring them back
rm -rf images/
rm images.tar.gz
```

Key points:
- The archive is referenced with its full path in `/staging`.
- The script removes extracted files before exiting so that only genuine output files are transferred back.
- If you only need specific outputs, also add `transfer_output_files = results.csv` to the submit file.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Check your disk usage

Log in to the CHTC submit server and determine:

1. How much space is your home directory currently using?
2. What are the three largest files or directories in your home directory?
3. If you have a `/staging` directory, how much space is it using?

:::::::::::::::: solution

```bash
# 1. Total home usage
du -sh /home/<netid>

# 2. Three largest items
du -h /home/<netid> --max-depth=1 | sort -rh | head -5

# 3. Staging usage (if applicable)
du -sh /staging/<netid>
```

If `du` takes a long time on your home directory, that itself may be a sign you have many files and should consider cleanup.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Best practices for data management on CHTC

### What goes where

| File type | Recommended location | Example |
|-----------|---------------------|---------|
| Submit files, scripts | `/home` | `train_model.sub`, `train_model.sh` |
| Small datasets (< 100 MB total) | `/home` (submit directory) | `titanic_train.csv` |
| Medium datasets (100 MB – few GB) | `/staging` | `image_dataset.tar.gz` |
| Large read-only reference data | SQUID | `pretrained_bert_model.tar.gz` |
| Job output logs | `/home` (automatically returned) | `job_12345.out` |
| Large output files | `/staging` (move after job completes) | `trained_model_weights.h5` |

### File size guidelines for `transfer_input_files`

- Individual files should ideally be **under a few GB**. Very large single files slow down transfers and can cause jobs to be held.
- If you have many small files (thousands of images, for example), **tar and compress** them into a single archive before transferring. HTCondor handles one large file much more efficiently than thousands of small ones.

```bash
# Package many small files into a single archive
tar -czf training_images.tar.gz images/

# In your job script, extract them
tar -xzf training_images.tar.gz
```

### Clean up after yourself

- Remove completed job logs you no longer need from `/home`.
- Delete intermediate data in `/staging` once you have your final results.
- Do not leave large files in SQUID indefinitely — remove them when your analysis campaign is complete.

::::::::::::::::::::::::::::::::::::: callout

#### CHTC data policies

- **No backups**: CHTC storage (including `/home`, `/staging`, and SQUID) is **not backed up**. Keep copies of irreplaceable data elsewhere (e.g., your department server, institutional storage, or a version control system for code).
- **Quotas are enforced**: Exceeding your quota can prevent you from submitting new jobs or receiving output files. Monitor your usage regularly.
- **Data retention**: Inactive data may be flagged for removal. If CHTC staff contact you about storage usage, respond promptly.
- **No guaranteed performance**: These are shared filesystems. Large transfers during peak times affect other users. Schedule bulk transfers during off-peak hours when possible.

For the latest policies, see the [CHTC data storage guide](https://chtc.cs.wisc.edu/uw-research-computing/file-avail-largedata).

::::::::::::::::::::::::::::::::::::::::::::::::

## Sensitive data considerations

CHTC general-purpose systems (`ap2002.chtc.wisc.edu`, etc.) are **not** designed for regulated or sensitive data. Specifically:

- Execute machines are shared among many users — your job files are not encrypted at rest on the execute node.
- Data in `/home`, `/staging`, and SQUID may be readable by system administrators.
- Network transfers within HTCondor are not encrypted by default.

If your research involves sensitive data (PHI, student records, proprietary datasets), contact the [CHTC facilitation team](https://chtc.cs.wisc.edu/uw-research-computing/get-help) **before** uploading any data. They can advise on options such as dedicated secure submit nodes or restricted pools.

## Comparison with cloud storage

If you have used cloud platforms before, the table below highlights the key differences:

| Feature | GCS (Google Cloud) | CHTC Storage |
|---------|-------------------|--------------|
| Cost | ~$0.02/GB/month + egress fees | Free (quota-based) |
| Access method | `gs://` URIs, APIs | File paths, HTCondor transfer |
| Scalability | Virtually unlimited (pay-as-you-go) | Limited by quotas (request increases) |
| Data durability | Highly redundant, 99.999999999% | No backups — user responsibility |
| Access control | IAM roles and policies | Unix file permissions |
| Sharing with jobs | SDK reads from bucket | `transfer_input_files` copies to job |
| Sensitive data | Configurable (VPC-SC, CMEK) | Not certified for regulated data (general systems) |

The biggest practical difference: on CHTC, your data must be **explicitly transferred** to each job. There is no shared filesystem that jobs can read from directly (unlike a GCS bucket that any authorized VM can access). This means you need to plan your data flow carefully — but it also means you never get a surprise bill.

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 3: Plan storage for a real project

You are starting a new image classification project with the following data:

- **Training images**: 15,000 JPEG files totaling 2.3 GB
- **Validation images**: 3,000 JPEG files totaling 450 MB
- **Pre-trained model weights**: 800 MB (read-only, used by all jobs)
- **Python training script**: 12 KB
- **Output**: each job produces a ~50 MB model file

Where would you store each component, and how would you structure your `transfer_input_files`?

:::::::::::::::: solution

1. **Training and validation images**: Package each set into a tar archive and store in `/staging`:
   ```bash
   tar -czf train_images.tar.gz train_images/
   tar -czf val_images.tar.gz val_images/
   # Move to staging
   mv train_images.tar.gz /staging/<netid>/
   mv val_images.tar.gz /staging/<netid>/
   ```

2. **Pre-trained model weights**: Since these are read-only and potentially shared across many jobs, SQUID is a good choice:
   ```bash
   cp pretrained_weights.h5 /squid/<netid>/
   ```

3. **Python script**: Small, stays in `/home` in your submit directory.

4. **Submit file:**
   ```
   transfer_input_files = /staging/<netid>/train_images.tar.gz, \
                          /staging/<netid>/val_images.tar.gz, \
                          http://proxy.chtc.wisc.edu/SQUID/<netid>/pretrained_weights.h5, \
                          train_model.py

   transfer_output_files = trained_model.pkl
   ```

5. **Output**: The `transfer_output_files` directive ensures only the 50 MB model file is returned, not the extracted images or temporary files.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- CHTC provides three storage tiers: `/home` (small files, code), `/staging` (larger datasets), and SQUID (large read-only data via HTTP).
- HTCondor copies files to and from jobs — use `transfer_input_files` and `transfer_output_files` in your submit file to control what is transferred.
- Storage on CHTC is free but quota-limited. Monitor your usage with `du -sh` and clean up after jobs complete.
- Package many small files into tar archives before transferring to improve efficiency.
- CHTC general-purpose systems are not certified for sensitive or regulated data — consult the facilitation team if your data has restrictions.
- Unlike cloud storage, there are no surprise bills — but there are also no backups, so keep copies of irreplaceable data elsewhere.

::::::::::::::::::::::::::::::::::::::::::::::::
