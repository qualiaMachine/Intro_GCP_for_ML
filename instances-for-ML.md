---
title: Compute for ML
---

This page provides guidance for selecting compute configurations in Google Cloud Platform (GCP) for machine learning workloads.  
While instance size is an important factor, effective performance depends on how you pair a machine type with optional GPU accelerators.

All pricing estimates are based on public rates for `us-central1` as of October 2025.  
Actual cost depends on sustained-use discounts, attached GPU quotas, and whether your project has promotional or educational credits.

### Key Terms

- **vCPU**: A *virtual CPU* represents one logical core allocated from a physical CPU. Two vCPUs typically correspond to one physical core on GCP hardware. More vCPUs allow for greater parallelism — useful when loading data, performing CPU-heavy preprocessing, or running multi-threaded operations. In GCP machine types, memory (RAM) generally scales with vCPUs — doubling vCPUs usually doubles available memory.  
- **Memory (GiB)**: System RAM available to the VM. Higher RAM supports larger batch sizes, data caching, and in-memory preprocessing, reducing disk I/O overhead.  
- **GPU (Graphics Processing Unit)**: Specialized hardware for parallel tensor operations used in deep learning model training and inference.  
- **Machine type**: Defines CPU and RAM resources; determines how many vCPUs and how much memory your instance has.  
- **Machine family**: A group of machine types optimized for a specific balance of performance, memory, and cost (e.g., `n2-standard-8`).  
- **Accelerator**: Optional hardware (such as GPUs or TPUs) that can be attached to certain VM families to speed up training and inference.  
- **Region**: The physical location of your compute resources (e.g., `us-central1`). Pricing and GPU availability can vary by region.  

### Reference Docs
- [Compute Engine VM Instance Pricing (applies to notebook backends)](https://cloud.google.com/compute/vm-instance-pricing)
- [Compute Engine GPU Pricing](https://cloud.google.com/compute/gpus-pricing)
- [All Compute Pricing Overview](https://cloud.google.com/compute/all-pricing)

### Key Concepts

- **Machine type vs. GPU**: The `machine_type` defines CPU and RAM resources — it is not a GPU by itself. You can attach a GPU by adding `accelerator_type` and `accelerator_count` (for example, `NVIDIA_L4` or `NVIDIA_TESLA_T4`). Only specialized machine families like `A2` include GPUs automatically.  
- **Full names and syntax**: Machine types follow the pattern `<family>-<series>-<vCPU count>`. For example:  
  - `n2-standard-8`: 8 vCPUs, 32 GB RAM  
  - `c2-standard-8`: 8 vCPUs, 32 GB RAM (CPU-optimized)  
  - `a2-highgpu-1g`: 12 vCPUs, 85 GB RAM, and 1 attached A100 GPU  
- **RAM requirements**: Minimum RAM should be at least 1.5× dataset size unless your workflow uses batching.  
- **Free tier**: Some smaller instance types (for example, `e2-micro`) may qualify for the [GCP Free Tier](https://cloud.google.com/free). Check usage limits before running persistent notebooks.

### Machine Families Overview

Different machine families are optimized for different workloads.  
Costs below are approximate per-hour rates for instances with 8 vCPUs in the `us-central1` region.

| Family | Optimized For | Example Machine Type | Approx. Cost/hr | Typical Model or Dataset Scale | Notes |
|---------|----------------|----------------------|-----------------|-------------------------------|-------|
| `E2` | General purpose | `e2-standard-8` | ~$0.25 | Small jobs or lightweight scripts | Cheapest option; slower CPUs |
| `N1` | Balanced compute (older gen) | `n1-standard-8` | ~$0.35 | Small to mid-sized ML (<100M params) | Broad GPU compatibility |
| `N2` | Balanced compute (newer gen) | `n2-standard-8` | ~$0.38 | Mid-sized ML and RAG pipelines (100M–500M params) | Common choice for notebooks |
| `C2` | Compute optimized | `c2-standard-8` | ~$0.45 | CPU-heavy preprocessing or feature extraction | High single-thread performance |
| `C3` | Next-gen compute optimized | `c3-standard-8` | ~$0.50 | High-performance CPU-only workloads | Faster I/O and networking |
| `A2` | GPU (A100) | `a2-highgpu-1g` | ~$2.93 (with 1×A100) | Large DL models (0.5B–10B params) | Fixed GPU counts, quota required |
| `A3` | GPU (H100) | `a3-highgpu-8g` | ~$32.00 (with 8×H100) | Transformer-scale models (10B–70B params) | High throughput, limited quota |
| `A4` | GPU (B200) | `a4-highgpu-4g` | ~$36.00 (with 4×B200) | Foundation models (70B+ params) | Highest-end, limited availability |
| `T2A` / `T2D` | Arm or AMD CPUs | `t2a-standard-8` | ~$0.20 | Low-cost inference or lightweight workloads | No GPU support |

**Cost notes:**  
- Prices vary by region and storage/network configuration.  
- `N2` instances are a typical choice for cost-effective ML workloads.  
- `A2–A4` families include GPUs by default; all others require attaching GPUs manually.

### Attaching GPUs vs. Using GPU Families

Attaching a GPU to a standard CPU family (`n1`, `n2`, or `c2`) is the most flexible and cost-efficient setup for research and medium-scale workloads.  
Dedicated GPU families like `A2`, `A3`, and `A4` are designed for very large or multi-GPU training but come with higher fixed costs and quota requirements.

| Approach | Best For | Pros | Cons |
|-----------|-----------|------|------|
| Attach GPU to Standard VM (`n1`/`n2` + `NVIDIA_L4`/`T4`) | Fine-tuning, RAG pipelines, and large-scale inference with models up to ~500M–1B params | Cheaper, flexible CPU/GPU balance, reusable for notebooks and jobs | Not ideal for multi-GPU scaling |
| Use GPU Machine Family (`A2`/`A3`/`A4`) | Multi-GPU training or high-throughput inference with models >1B params | High throughput, optimized GPU interconnects | Expensive, quota-restricted, fixed GPU count |

For large-scale RAG deployments using very large models (e.g., 7B–70B parameters), `A2` or `A3` instances may be required to hold the model in GPU memory during inference.  
However, when using model sharding or quantized models under 20–40 GB total, attached L4 GPUs on `n2` machines remain cost-effective.

### Typical GPU Options for Attached Configurations

| GPU Type | CUDA Version | Approx. Price/hr | Model Size Range | Dataset Scale | System RAM (Recommended) | Typical Use |
|-----------|--------------|------------------|------------------|----------------|---------------------------|--------------|
| `NVIDIA_TESLA_T4` | CUDA 11.x–12.x | ~$0.35 | ≤100 M params | ≤10 GB | ≥16 GB | Entry GPU for CNNs, small transformers |
| `NVIDIA_L4` | CUDA 12.x | ~$0.60 | ≤500 M–1 B params | ≤50 GB | ≥32 GB | Moderate training, RAG inference, fine-tuning |
| `NVIDIA_TESLA_V100` | CUDA 11.x | ~$2.48 | 0.5 B–2 B params | ≤100 GB | ≥64 GB | High-performance deep learning |
| `NVIDIA_A100_40GB` | CUDA 11.x–12.x | ~$2.93 | 2 B–10 B params | ≤200 GB | ≥128 GB | Research-scale model training |
| `NVIDIA_H100` | CUDA 12.x | ~$4.00 | 10 B–70 B params | ≤500 GB | ≥256 GB | Transformer and LLM training/inference |
| `NVIDIA_B200` | CUDA 12.x | ~$5.00+ | >70 B params | ≥1 TB | ≥512 GB | Foundation-model or multi-node workloads |

### Example Workload Choices

- **RAG with LLMs:** Retrieval-augmented generation pipelines rely mainly on CPU and memory for vector retrieval and embedding operations, with moderate GPU usage during inference. Recommended: `n2-standard-8` + `NVIDIA_L4` for typical RAG; move to `a2-highgpu-1g` or `a3-highgpu` if the model exceeds 1B parameters or GPU memory limits.  
- **Training a 100M-parameter neural network:** This model size fits comfortably on a single mid-tier GPU and benefits from faster GPU memory bandwidth. Recommended: `n1-standard-8` + `NVIDIA_TESLA_T4` for affordability, or `NVIDIA_L4` if training time matters more than cost.  
- **Multi-GPU or LLM fine-tuning (billions of parameters):** Large models (1B–70B parameters) often require multiple A100, H100, or B200 GPUs in parallel. Recommended: `a2-highgpu-2g` (2×A100) or larger depending on model size and parallelism. Cost note: Fine-tuning billion-parameter models can easily exceed $200–$500 per hour of GPU time. Even short fine-tunes may consume hundreds of dollars in credits. Plan carefully, monitor utilization, and test your pipeline with smaller models first.

### Example Configurations

| Dataset Size | Recommended Notebook Instance | vCPU | Memory (GiB) | GPU / Accelerator | Price/hr (USD) | Typical Use |
|---------------|--------------------------------|------|----------------|-------------------|----------------|--------------|
| < 1 GB | `e2-micro` (Free Tier) | 2 | 1 | None | Free Tier | Lightweight code tests |
| < 1 GB | `n2-standard-4` | 4 | 16 | None | ~$0.17 | Preprocessing, regression, small models |
| < 1 GB | `n1-standard-8` + `NVIDIA_TESLA_T4` | 8 | 30 | 1× T4 | ~$0.55 | Entry GPU runs, small CNNs |
| 10 GB | `c2-standard-8` | 8 | 32 | None | ~$0.34 | CPU-heavy ML tasks |
| 10 GB | `n2-standard-8` + `NVIDIA_L4` | 8 | 32 | 1× L4 | ~$0.75 | Moderate deep learning workloads |
| 50 GB | `a2-highgpu-2g` (2× A100) | 24 | 170 | 2× A100 | ~$5.90 | Multi-GPU training, large-model inference |
| 100 GB | `a3-highgpu-8g` (8× H100) | 128 | 512 | 8× H100 | ~$32.00 | Transformer or LLM fine-tuning |
| 1 TB+ | `a4-highgpu-4g` (4× B200) | 96 | 768 | 4× B200 | ~$36.00 | Foundation-model scale training |

### General Notes

- For small datasets, CPUs are often faster to start and cheaper to run.  
- When moving from CPU to GPU training, keep the same script and simply change:
  - `container_uri` to a GPU-enabled image (for example, `pytorch-gpu.*`)
  - Add `accelerator_type` and `accelerator_count` in your `CustomTrainingJob`.

### Summary

1. Choose the `machine_type` for CPU and memory resources.  
2. Attach a GPU with `accelerator_type` and `accelerator_count` if needed.  
3. Only `A2`, `A3`, and `A4` families include GPUs automatically.  
4. For most research training jobs, `n1-standard-8` + `NVIDIA_TESLA_T4` or `NVIDIA_L4` is a practical and affordable starting point.  
5. Fine-tuning or large-scale inference with billion-parameter models can be extremely expensive; validate your workflow with smaller models first.
