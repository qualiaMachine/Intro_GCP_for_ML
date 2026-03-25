---
title: Compute for ML
---

This page provides guidance for selecting compute configurations on CHTC for machine learning workloads. CHTC resources are free for UW-Madison researchers, so the focus is on choosing the right hardware for your workload rather than minimizing cost.

### Reference Docs

- [CHTC GPU Lab Guide](https://chtc.cs.wisc.edu/uw-research-computing/gpu-lab)
- [CHTC Machine Learning Guide](https://chtc.cs.wisc.edu/uw-research-computing/machine-learning-htc)
- [HTCondor Manual — Submitting Jobs](https://htcondor.readthedocs.io/en/latest/users-manual/submitting-a-job.html)

### Key Terms

- **CPU**: A general-purpose processor. Most ML preprocessing, feature engineering, and tree-based models (XGBoost, random forests) run efficiently on CPUs.
- **GPU (Graphics Processing Unit)**: Specialized hardware for parallel tensor operations used in deep learning model training and inference. CHTC's GPU Lab provides access to high-end NVIDIA GPUs.
- **Memory (RAM)**: System memory available to your job. Higher RAM supports larger batch sizes, data caching, and in-memory preprocessing.
- **Disk**: Local scratch storage on the execute node. Your input files are transferred here, and outputs are written here during the job.
- **Execute node**: The machine where your HTCondor job runs. You request resources (CPUs, memory, disk, GPUs) in your submit file, and HTCondor matches your job to a machine that meets those requirements.

### Key Concepts

- **Right-sizing matters even when it's free.** Over-requesting resources (e.g., 8 CPUs when you need 1, or 32 GB RAM when you need 4 GB) means your job waits longer in the queue because fewer machines can match your request. Always start small and scale up based on actual usage.
- **Check actual resource usage** after your first job completes. The job log file reports actual memory and disk usage. Use this to refine future requests.
- **GPU ≠ always faster.** For small datasets (< 10,000 rows) or simple models (< 1M parameters), CPU training is often faster end-to-end because GPU jobs have provisioning and data transfer overhead.

### Available GPUs on CHTC

CHTC's GPU Lab includes several GPU types. You can request specific hardware using `require_gpus` in your submit file.

| GPU Type | VRAM | Best For | require_gpus filter |
|----------|------|----------|-------------------|
| NVIDIA T4 | 16 GB | Entry-level deep learning, small transformers | `(GlobalMemoryMb >= 15000)` |
| NVIDIA L40 | 48 GB | Medium models, RAG inference, fine-tuning | `(GlobalMemoryMb >= 40000)` |
| NVIDIA A100 (40 GB) | 40 GB | Large deep learning, multi-billion param models | `(GlobalMemoryMb >= 35000)` |
| NVIDIA A100 (80 GB) | 80 GB | Very large models, large batch training | `(GlobalMemoryMb >= 75000)` |
| NVIDIA H100 | 80 GB | Transformer-scale training, LLM fine-tuning | `(GlobalMemoryMb >= 75000)` |
| NVIDIA H200 | 141 GB | Frontier models, 70B+ parameter fine-tuning | `(GlobalMemoryMb >= 130000)` |

### GPU Job Runtime Limits

CHTC GPU Lab jobs have runtime limits based on a job length category you specify:

| Category | Max Runtime | Submit file setting |
|----------|------------|-------------------|
| Short | 12 hours | `+GPUJobLength = "short"` |
| Medium | 24 hours | `+GPUJobLength = "medium"` |
| Long | 7 days | `+GPUJobLength = "long"` |

**Tip:** Short jobs get scheduled faster because more machines accept them. Start with "short" and only increase if your training genuinely needs more time. For very long training runs, implement checkpointing so you can resume if a job is interrupted.

### Example Resource Requests

| Workload | CPUs | Memory | Disk | GPUs | Notes |
|----------|------|--------|------|------|-------|
| XGBoost on Titanic (< 1 GB data) | 1 | 2 GB | 1 GB | 0 | CPU is sufficient |
| Small neural network (< 10M params) | 1 | 4 GB | 2 GB | 0 | CPU often faster for small models |
| Medium neural network (10M–500M params) | 1 | 8 GB | 4 GB | 1 (T4 or L40) | GPU speeds up training significantly |
| Large transformer fine-tuning (500M–10B) | 4 | 32 GB | 20 GB | 1 (A100 40/80 GB) | Need large VRAM for model weights |
| Very large model (10B–70B) | 8 | 64 GB | 50 GB | 1 (H100 or H200) | Quantization may be needed |
| Hyperparameter sweep (many small jobs) | 1 | 4 GB | 2 GB | 0 or 1 | Submit many jobs in parallel |

### Example Submit File Snippets

**CPU-only job:**
```
request_cpus = 1
request_memory = 4GB
request_disk = 2GB
```

**Single GPU job (any GPU with >= 16 GB VRAM):**
```
request_cpus = 1
request_gpus = 1
require_gpus = (GlobalMemoryMb >= 15000)
request_memory = 8GB
request_disk = 4GB
+WantGPULab = true
+GPUJobLength = "short"
```

**Single GPU job (A100 or better):**
```
request_cpus = 4
request_gpus = 1
require_gpus = (GlobalMemoryMb >= 35000)
request_memory = 32GB
request_disk = 20GB
+WantGPULab = true
+GPUJobLength = "medium"
```

### When Does Model Size Justify a GPU?

| Model scale | Parameters | Example models | Recommended hardware |
|-------------|-----------|----------------|---------------------|
| Small | < 10M | Logistic regression, small CNNs, XGBoost | CPU only |
| Medium | 10M–500M | ResNets, BERT-base, mid-sized transformers | T4 or L40 |
| Large | 500M–10B | GPT-2, LLaMA-7B, fine-tuning large transformers | A100 (40/80 GB) |
| Very large | 10B–70B | LLaMA-70B, Mixtral | H100 or H200 |
| Frontier | 70B+ | GPT-4-scale, multi-expert models | Cloud (multi-node) |

### General Guidelines

1. **Start small.** Request minimal resources for your first run. Check the job log to see actual usage, then adjust.
2. **CPU first, GPU second.** Only add a GPU when your model and data are large enough to benefit from it.
3. **Match your container to your hardware.** Use a CUDA-enabled Docker image (e.g., `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`) for GPU jobs.
4. **Use short job lengths.** Short jobs get scheduled faster. Only request longer runtimes if genuinely needed.
5. **Implement checkpointing** for training runs longer than a few hours. This protects against job eviction and lets you resume training across multiple job submissions.
6. **Don't over-request.** Requesting 8 CPUs and 64 GB RAM for a job that uses 1 CPU and 2 GB RAM wastes shared resources and increases your queue wait time.
