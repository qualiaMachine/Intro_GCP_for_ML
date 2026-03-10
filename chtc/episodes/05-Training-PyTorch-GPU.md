---
title: "PyTorch GPU Training on CHTC"
teaching: 20
exercises: 15
---

::::::::::::::::::::::::::::::::::::: questions

- How do I request GPUs for an HTCondor job?
- How does a GPU training job differ from a CPU job in HTCondor?
- What GPU hardware is available on CHTC?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Prepare `.npz` data for PyTorch training.
- Write HTCondor submit files for both CPU and GPU training jobs.
- Request GPUs and optionally specify GPU requirements.
- Compare CPU vs. GPU training times.

::::::::::::::::::::::::::::::::::::::::::::::::

## Overview

Requesting a GPU on CHTC is simple: add `request_gpus = 1` to your submit file. HTCondor finds a machine with a GPU, runs your job inside a container with CUDA support, and transfers the results back. PyTorch's `torch.cuda.is_available()` automatically detects the GPU — no manual CUDA configuration needed.

## Step 1: Prepare the data

The PyTorch trainer expects `.npz` files (NumPy compressed arrays) rather than raw CSV. Run the data preparation script on the submit node:

```bash
cd /home/$USER/workshop/

# Prepare train/val splits as .npz files
python3 prepare_data.py --input titanic_train.csv --output_train train_data.npz --output_val val_data.npz
```

This creates two files:
- `train_data.npz` — training features and labels
- `val_data.npz` — validation features and labels

This is a lightweight operation (the Titanic dataset is small), so running it on the submit node is fine.

## Step 2: The training script

The `train_nn.py` script defines a small neural network (`TitanicNet`) and trains it with early stopping. Key features:

- **`TitanicNet`** — 3-layer network (64 → 32 → 1) with ReLU and Sigmoid
- **Early stopping** — stops training when validation loss stops improving
- **Metrics output** — saves `metrics.json` with final accuracy, loss, and hyperparameters (we'll use this in Episode 6 for HP tuning)
- **Reproducibility** — fixed random seeds for consistent results

Key code that makes GPU detection automatic:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

When your job runs on a CHTC worker with a GPU (because you requested one), `torch.cuda.is_available()` returns `True` automatically. No CUDA path configuration needed — the container handles it.

## Step 3: CPU submit file

```
# train_nn_cpu.sub — PyTorch training on CPU only

universe     = vanilla
executable   = run_nn.sh

log          = nn_cpu_$(Cluster).log
output       = nn_cpu_$(Cluster).out
error        = nn_cpu_$(Cluster).err

request_cpus   = 1
request_memory = 2GB
request_disk   = 2GB

transfer_input_files = train_nn.py, run_nn.sh, train_data.npz, val_data.npz
transfer_output_files = model.pt, metrics.json, eval_history.csv

container_image = docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

arguments = --epochs 500 --learning_rate 0.001 --patience 20

queue 1
```

## Step 4: GPU submit file

The GPU version is nearly identical — just add `request_gpus`:

```
# train_nn_gpu.sub — PyTorch training on GPU

universe     = vanilla
executable   = run_nn.sh

log          = nn_gpu_$(Cluster).log
output       = nn_gpu_$(Cluster).out
error        = nn_gpu_$(Cluster).err

request_gpus   = 1
request_cpus   = 1
request_memory = 4GB
request_disk   = 2GB

transfer_input_files = train_nn.py, run_nn.sh, train_data.npz, val_data.npz
transfer_output_files = model.pt, metrics.json, eval_history.csv

container_image = docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

arguments = --epochs 500 --learning_rate 0.001 --patience 20

queue 1
```

**The only differences from the CPU version:**
- `request_gpus = 1` — tells HTCondor to find a machine with a GPU
- `request_memory = 4GB` — GPU machines typically have more memory available

## Step 5: Submit and compare

```bash
# Submit both jobs
condor_submit train_nn_cpu.sub
condor_submit train_nn_gpu.sub

# Monitor
condor_q
```

After both complete, compare:

```bash
# Check which device each job used
grep "Using device" nn_cpu_*.out
grep "Using device" nn_gpu_*.out

# Compare training times (look at HTCondor log for wall clock time)
grep "Job terminated" nn_cpu_*.log
grep "Job terminated" nn_gpu_*.log

# Compare final metrics
cat metrics.json  # (from whichever job you want to inspect)
```

## Requesting specific GPUs

CHTC's GPU Lab includes several GPU types. You can be specific about what you need:

```
# Require at least 40 GB GPU memory (matches A100-40GB, A100-80GB, H100, H200)
require_gpus = (GlobalMemoryMb >= 40000)

# Require a specific GPU capability level
require_gpus = (Capability >= 8.0)
```

GPU capability levels on CHTC:
| GPU | Memory | Capability | Typical use |
|-----|--------|-----------|-------------|
| A100 40GB | 40 GB | 8.0 | Large model training, fine-tuning |
| A100 80GB | 80 GB | 8.0 | Very large models |
| H100 | 80 GB | 9.0 | Frontier training |
| H200 | 141 GB | 9.0 | Largest single-GPU workloads |

For the Titanic dataset, any GPU will do — the model is tiny. But for larger models, specifying GPU requirements ensures your job lands on appropriate hardware.

::::::::::::::::::::::::::::::::::::: callout

### GPU queue wait times

GPU jobs may wait longer than CPU jobs because GPU machines are in higher demand. Check estimated wait times:

```bash
condor_q -better-analyze <cluster_id>
```

For time-sensitive work, consider starting with CPU training for debugging, then switching to GPU for the final run.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: callout

### Job time limits

CHTC GPU jobs have time limits:
- **Short jobs:** up to 12 hours
- **Medium jobs:** up to 24 hours
- **Long jobs:** up to 7 days

For deep learning models that need more than 12 hours, implement **checkpointing** — save the model state periodically and resume from the latest checkpoint if the job is evicted. The Titanic model trains in seconds, but this matters for larger models.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: CPU vs. GPU timing

1. Submit both CPU and GPU training jobs for the Titanic neural network.
2. Compare the wall-clock times from the `.log` files.
3. For this tiny dataset, is there a meaningful difference? When would the GPU advantage become significant?

:::::::::::::::: solution

For the Titanic dataset (~700 training rows, tiny model), training is so fast that GPU overhead (data transfer to GPU, kernel launches) may make the GPU version *slower* than CPU. The GPU advantage becomes significant when:

- The dataset has thousands+ of rows
- The model has millions+ of parameters
- Each epoch involves substantial matrix operations
- You're training for hundreds+ of epochs

For real-world deep learning (images, NLP, large datasets), GPUs provide 10–100x speedups.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Require specific GPU hardware

Write a submit file that requires an A100 GPU with at least 80 GB of memory. What `require_gpus` line would you use?

:::::::::::::::: solution

```
require_gpus = (GlobalMemoryMb >= 80000)
```

This matches A100-80GB, H100, and H200 GPUs. Note that being more specific about GPU requirements may increase queue wait times, since fewer machines match your request.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Adding `request_gpus = 1` to your submit file is all it takes to request GPU hardware on CHTC.
- PyTorch's `torch.cuda.is_available()` automatically detects CHTC GPUs — no manual CUDA configuration needed.
- Use `require_gpus` to request specific GPU capabilities (memory, compute capability).
- CHTC GPUs (A100, H100, H200) are free to UW-Madison researchers.
- Always test on CPU first for debugging, then switch to GPU for production training.

::::::::::::::::::::::::::::::::::::::::::::::::
