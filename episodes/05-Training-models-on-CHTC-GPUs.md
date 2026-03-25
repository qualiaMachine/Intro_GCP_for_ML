---
title: "Training Models on CHTC GPUs"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions

- How do you request and use GPUs for PyTorch training on CHTC?
- What GPU hardware is available in the CHTC GPU Lab, and how do you select the right one?
- When is GPU training worth the overhead compared to CPU-only training?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Prepare the Titanic dataset and save train/val arrays to compressed `.npz` files on the submit node.
- Write an HTCondor submit file that requests GPU resources and uses a Docker container with PyTorch and CUDA.
- Submit a GPU training job to the CHTC GPU Lab and monitor its progress.
- Compare CPU vs. GPU training times and understand when GPU acceleration is beneficial.

::::::::::::::::::::::::::::::::::::::::::::::::

## Overview

In the previous episode, we trained an XGBoost model using HTCondor on CHTC. Here, we'll do the same for a PyTorch neural network -- but this time we'll use GPUs. The CHTC GPU Lab provides shared access to high-end NVIDIA GPUs for researchers at UW-Madison. We'll learn how to request GPU resources in an HTCondor submit file, use a GPU-enabled Docker container, and understand when GPUs actually help.

## The CHTC GPU Lab

CHTC maintains a dedicated GPU Lab with several types of NVIDIA GPUs available to researchers. The current inventory includes:

| GPU Model | VRAM | Typical Use Cases |
|-----------|------|-------------------|
| NVIDIA T4 | 16 GB | Inference, small-to-medium training |
| NVIDIA L40 | 48 GB | Medium training, graphics workloads |
| NVIDIA A100 (40 GB) | 40 GB | Large-scale training, mixed precision |
| NVIDIA A100 (80 GB) | 80 GB | Large models, large batch sizes |
| NVIDIA H100 | 80 GB | Cutting-edge training, transformer models |
| NVIDIA H200 | 141 GB | Very large models, high-memory workloads |

::::::::::::::::::::::::::::::::::::: callout

### GPU Lab access

To use the GPU Lab, your HTCondor submit file must include `+WantGPULab = true`. Without this flag, your job will not be matched to GPU Lab hardware. All CHTC users with an active account can access the GPU Lab -- no special quota request is needed (unlike some cloud providers).

:::::::::::::::::::::::::::::::::::::

## Prepare data as `.npz`

Unlike the XGBoost script from Episode 4 (which handles preprocessing internally from raw CSV), our PyTorch script expects pre-processed NumPy arrays. We prepare those on the submit node and save them as `.npz` files.

Why `.npz`? NumPy's `.npz` files are compressed binary containers that can store multiple arrays (e.g., features and labels) together in a single file:

- **Compact and fast:** smaller than CSV, and one file can hold multiple arrays (`X_train`, `y_train`).
- **Transfer-friendly:** each `.npz` is a single file -- one transfer operation instead of streaming many small files.
- **Reproducible:** unlike CSV, `.npz` preserves exact dtypes and shapes across environments.

Run the following Python script on the submit node to create the data files. Save this as `prepare_data.py`:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Titanic CSV
df = pd.read_csv("titanic_train.csv")

# Minimal preprocessing to numeric arrays
sex_enc = LabelEncoder().fit(df["Sex"])
df["Sex"] = sex_enc.transform(df["Sex"])
df["Embarked"] = df["Embarked"].fillna("S")
emb_enc = LabelEncoder().fit(df["Embarked"])
df["Embarked"] = emb_enc.transform(df["Embarked"])
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values
y = df["Survived"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

np.savez("train_data.npz", X_train=X_train, y_train=y_train)
np.savez("val_data.npz", X_val=X_val, y_val=y_val)

print(f"Created train_data.npz ({X_train.shape[0]} samples) "
      f"and val_data.npz ({X_val.shape[0]} samples)")
```

Run this on the submit node:

```bash
python3 prepare_data.py
```

You should now have `train_data.npz` and `val_data.npz` in your working directory.

## The training script: `train_nn.py`

Find this file in our repo: `Intro_GCP_for_ML/scripts/train_nn.py`. It does three things:

1. Loads `.npz` files from local paths.
2. Trains a small neural network (a 3-layer MLP) with early stopping.
3. Writes all outputs side-by-side: `model.pt`, `metrics.json`, `eval_history.csv`, and `training.log`.

::::::::::::::::::::::::::::::::::::: callout

### What's inside `train_nn.py`? (Quick reference)

You don't need to understand every line of the PyTorch code for this workshop -- the focus is on how to package and run *any* training script on CHTC GPUs. Here is a quick orientation:

- **Model**: A small feedforward network (`TitanicNet`) -- the architecture details are not important for this lesson.
- **Early stopping**: Training halts when validation loss stops improving (controlled by `--patience`). This saves compute time.
- **Device detection**: The script automatically detects whether a GPU is available (`torch.cuda.is_available()`) and moves the model and data accordingly. The same script works on both CPU and GPU without modification.

:::::::::::::::::::::::::::::::::::::

## The wrapper script: `run_training.sh`

When using HTCondor's Docker universe, the executable must be a shell script that runs inside the container. Create a file called `run_training.sh`:

```bash
#!/bin/bash

python3 train_nn.py \
    --train train_data.npz \
    --val val_data.npz \
    --epochs 500 \
    --learning_rate 0.001 \
    --patience 50
```

Make it executable:

```bash
chmod +x run_training.sh
```

::::::::::::::::::::::::::::::::::::: callout

### Why a wrapper script?

In HTCondor's Docker universe, the `executable` field specifies a script that runs inside the container. You cannot directly set `executable = python3` with arguments in the same way you might on the command line. The wrapper script provides a clean way to call Python with all the necessary arguments. This pattern is standard practice for CHTC Docker jobs.

:::::::::::::::::::::::::::::::::::::

## The GPU submit file

Here is the HTCondor submit file for running our PyTorch training on a GPU. Save this as `gpu_train.sub`:

```
universe = docker
docker_image = pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

executable = run_training.sh

transfer_input_files = train_nn.py, train_data.npz, val_data.npz

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

log    = gpu_job_$(Cluster).log
output = gpu_job_$(Cluster).out
error  = gpu_job_$(Cluster).err

request_cpus   = 1
request_gpus   = 1
require_gpus   = (GlobalMemoryMb >= 10000)
request_memory = 8GB
request_disk   = 4GB

+WantGPULab  = true
+GPUJobLength = "short"

queue 1
```

Let's break down the key GPU-specific lines:

### `universe = docker` and `docker_image`

We use a prebuilt PyTorch Docker image that includes CUDA and cuDNN. This means we don't need to install any GPU drivers or libraries ourselves -- the container has everything PyTorch needs to use the GPU.

### `request_gpus = 1`

This tells HTCondor to match your job to a machine that has at least one available GPU and to allocate that GPU to your job.

### `require_gpus = (GlobalMemoryMb >= 10000)`

This is a constraint expression that filters GPU hardware. Here we require at least 10 GB of GPU memory, which excludes older/smaller GPUs. You can also constrain on specific GPU properties:

- `(GlobalMemoryMb >= 40000)` -- require 40+ GB (matches A100 40GB and above)
- `(GlobalMemoryMb >= 80000)` -- require 80+ GB (matches A100 80GB, H100, H200)

### `+WantGPULab = true`

This flag tells HTCondor to route your job to the CHTC GPU Lab pool. Without it, your job will not be matched to GPU Lab machines.

### `+GPUJobLength`

CHTC GPU jobs have runtime limits to ensure fair sharing. You must declare your expected job length:

| Value | Maximum Runtime |
|-------|----------------|
| `"short"` | 12 hours |
| `"medium"` | 24 hours |
| `"long"` | 7 days |

Choose the shortest category that fits your job. Shorter jobs are scheduled faster because they can fill smaller gaps in the schedule. If your job exceeds the declared time limit, it will be held or evicted.

## Submit and monitor the GPU job

Submit the job:

```bash
condor_submit gpu_train.sub
```

Monitor with standard HTCondor commands:

```bash
# Check job status
condor_q

# Watch job status update every 5 seconds
condor_q -nobatch

# View detailed job information
condor_q -l <job_id>

# Check which GPU was assigned (after job starts running)
condor_q -af GPUs_DeviceName GPUs_GlobalMemoryMb
```

::::::::::::::::::::::::::::::::::::: callout

### GPU job queue times

GPU jobs may wait longer in the queue than CPU-only jobs because GPU hardware is a shared, limited resource. Jobs requesting `"short"` runtimes generally start sooner because they can backfill into smaller scheduling gaps. If your job is idle for a long time, check that your `require_gpus` constraint is not too restrictive.

:::::::::::::::::::::::::::::::::::::

Once the job completes, check the output:

```bash
# View training output
cat gpu_job_*.out

# Check for errors
cat gpu_job_*.err

# List output files
ls -la model.pt metrics.json eval_history.csv training.log
```

## Comparing CPU vs. GPU training

To compare, you can also submit a CPU-only version of the job. Create `cpu_train.sub`:

```
universe = docker
docker_image = pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

executable = run_training.sh

transfer_input_files = train_nn.py, train_data.npz, val_data.npz

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

log    = cpu_job_$(Cluster).log
output = cpu_job_$(Cluster).out
error  = cpu_job_$(Cluster).err

request_cpus   = 1
request_memory = 8GB
request_disk   = 4GB

queue 1
```

Notice that the CPU version simply omits `request_gpus`, `require_gpus`, `+WantGPULab`, and `+GPUJobLength`. The same Docker image works for both -- PyTorch will automatically fall back to CPU when no GPU is available.

Submit and compare:

```bash
condor_submit cpu_train.sub
```

For the Titanic dataset (roughly 700 training samples, 7 features, 3-layer MLP), you will likely find that:

- **CPU training time**: a few seconds
- **GPU training time**: similar or slightly longer

The GPU overhead (CUDA initialization, data transfer to GPU memory) can actually make small jobs *slower* on a GPU. This is expected and normal.

:::::::::::::::::::::::::::::::::::::::: challenge

### When is a GPU worth it?

Consider the following scenarios. For each one, decide whether you would request a GPU or stick with CPU-only training on CHTC:

1. Training a 3-layer MLP on the Titanic dataset (891 rows, 7 features).
2. Fine-tuning a ResNet-50 model on 50,000 images (224x224 pixels).
3. Training a transformer language model with 125 million parameters on 10 GB of text.
4. Running 200 independent hyperparameter trials of a small random forest on tabular data.

:::::::::::::::::::::::::::::::::::::::: solution

### Solution

1. **CPU.** The dataset and model are tiny. GPU overhead would likely make it slower, and you would wait longer in the queue for a GPU slot.
2. **GPU.** Image models like ResNet involve large matrix operations (convolutions) on high-dimensional data. A GPU will be significantly faster -- potentially 10-50x.
3. **GPU (possibly multiple).** Transformer training is extremely compute-intensive. Even a single A100 might take days; without a GPU this would be impractical.
4. **CPU.** Random forests are not GPU-accelerated in standard scikit-learn. The 200 trials are independent, so submit them as 200 separate CPU jobs and let HTCondor parallelize across the cluster.

:::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::::

## GPU training workflow summary

Here is the complete workflow for GPU training on CHTC:

1. **Prepare data** on the submit node (create `.npz` files).
2. **Write your training script** (`train_nn.py`) to detect GPU automatically with `torch.cuda.is_available()`.
3. **Write a wrapper script** (`run_training.sh`) that calls Python inside the container.
4. **Write a submit file** (`gpu_train.sub`) with `request_gpus`, `require_gpus`, `+WantGPULab = true`, and `+GPUJobLength`.
5. **Submit and monitor** with `condor_submit` and `condor_q`.
6. **Collect results** from the transferred output files.

:::::::::::::::::::::::::::::::::::::::: challenge

### Modify the submit file

Starting from the GPU submit file above, make the following changes:

1. Request a GPU with at least 40 GB of memory.
2. Set the job length to "medium" (24-hour limit).
3. Request 16 GB of system memory instead of 8 GB.

:::::::::::::::::::::::::::::::::::::::: solution

### Solution

The three lines that change:

```
require_gpus   = (GlobalMemoryMb >= 40000)
request_memory = 16GB
+GPUJobLength  = "medium"
```

This would match A100 (40 GB or 80 GB), H100, or H200 GPUs, allow up to 24 hours of runtime, and provide more system RAM for data loading and preprocessing.

:::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::::

## Additional resources

- [CHTC GPU Lab guide](https://chtc.cs.wisc.edu/uw-research-computing/gpu-lab)
- [HTCondor documentation on GPUs](https://htcondor.readthedocs.io/en/latest/)
- [PyTorch CUDA documentation](https://pytorch.org/docs/stable/cuda.html)
- [Docker Hub: PyTorch images](https://hub.docker.com/r/pytorch/pytorch/tags)

::::::::::::::::::::::::::::::::::::: keypoints

- Use `request_gpus`, `require_gpus`, `+WantGPULab = true`, and `+GPUJobLength` in your HTCondor submit file to request GPU resources from the CHTC GPU Lab.
- GPU-enabled Docker containers (e.g., `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`) provide CUDA and cuDNN so your code can use the GPU without manual driver installation.
- Prepare data as `.npz` files on the submit node and transfer them with the job -- this is compact, fast, and reproducible.
- GPU acceleration pays off for large models and large datasets; for small problems like Titanic, CPU is often faster due to GPU initialization overhead.
- Declare the shortest `+GPUJobLength` that fits your job (`"short"` = 12 hr, `"medium"` = 24 hr, `"long"` = 7 days) -- shorter jobs are scheduled sooner.

::::::::::::::::::::::::::::::::::::::::::::::::
