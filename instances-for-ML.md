---
title: Compute for ML
---

This page provides guidance for selecting compute configurations in Google Cloud Platform (GCP) for machine learning workloads.  
While instance size is an important factor, effective performance depends on how you pair a machine type with optional GPU accelerators.

**All pricing estimates** are based on public rates for `us-central1` as of October 2025.  
Actual cost depends on sustained-use discounts, attached GPU quotas, and whether your project has promotional or educational credits.

### Reference Docs
- [Compute Engine VM Instance Pricing (applies to notebook backends)](https://cloud.google.com/compute/vm-instance-pricing)
- [Compute Engine GPU Pricing](https://cloud.google.com/compute/gpus-pricing)
- [All Compute Pricing Overview](https://cloud.google.com/compute/all-pricing)

### Key Concepts

- **Machine type vs. GPU**: The `machine_type` defines CPU and RAM resources — it is *not* a GPU by itself.  
  You can attach a GPU by adding `accelerator_type` and `accelerator_count` (for example, `NVIDIA_TESLA_T4` or `NVIDIA_L4`).  
  Only specialized machine families like `A2` include GPUs automatically.

- **CPU-focused series**: `N2` and `C2` are optimized for CPU-heavy workloads such as preprocessing, feature engineering, or classical ML.

- **GPU-focused series**: GPUs handle deep learning workloads. You can attach them to standard families (for example, `n1-standard-8` + `NVIDIA_TESLA_T4`) or use dedicated GPU machine families (`A2`, `A3`, `A4`).

- **RAM requirements**: Minimum RAM should be at least 1.5× dataset size unless your workflow uses batching.

- **Free tier**: Some smaller instance types (for example, `e2-micro`) may qualify for the [GCP Free Tier](https://cloud.google.com/free). Check usage limits before running persistent notebooks.

### Common Accelerator Choices

| Accelerator Type | CUDA Version | Approx. Price/hr | Typical Use |
|------------------|--------------|------------------|-------------|
| `NVIDIA_TESLA_T4` | CUDA 11.x–12.x | ~$0.35 | Cost-effective entry GPU for small models |
| `NVIDIA_L4` | CUDA 12.x | ~$0.60 | Balanced option for mid-scale DL |
| `NVIDIA_TESLA_V100` | CUDA 11.x | ~$2.48 | High-performance training (vision, NLP) |
| `NVIDIA_A100_40GB` | CUDA 11.x–12.x | ~$2.93 | Research-scale deep learning workloads |
| `NVIDIA_H100` | CUDA 12.x | ~$4.00 | Transformer-scale model training |
| `NVIDIA_B200` | CUDA 12.x | ~$5.00+ | Large foundation models and experimental work |

### General Notes

- For small datasets, CPUs are often faster to start and cheaper to run.  
- For deep learning, attach a GPU instead of switching the entire machine family.  
- To move from CPU → GPU training, keep the same script and simply change:
  - `container_uri` to a GPU-enabled PyTorch image (`pytorch-gpu.*`)
  - Add `accelerator_type` and `accelerator_count` in your `CustomTrainingJob`.

### Example Configurations

| **Dataset Size** | **Recommended Notebook Instance** | **vCPU** | **Memory (GiB)** | **GPU / Accelerator** | **Price/hr (USD)** | **Typical Use** |
|------------------|-----------------------------------|-----------|------------------|-----------------------|-------------------|-----------------|
| < 1 GB | `e2-micro` *(Free Tier)* | 2 | 1 | None | Free Tier | Lightweight code tests |
| < 1 GB | `n2-standard-4` | 4 | 16 | None | ~$0.17 | Preprocessing, regression, small models |
| < 1 GB | `n1-standard-8` + `NVIDIA_TESLA_T4` | 8 | 30 | 1× T4 | ~$0.55 | Entry GPU runs, small CNNs |
| 10 GB | `c2-standard-8` | 8 | 32 | None | ~$0.34 | CPU-heavy ML tasks |
| 10 GB | `n2-standard-8` + `NVIDIA_L4` | 8 | 32 | 1× L4 | ~$0.75 | Moderate deep learning workloads |
| 50 GB | `a2-highgpu-2g` *(2× A100)* | 24 | 170 | 2× A100 | ~$5.90 | Multi-GPU training, research workloads |
| 100 GB | `a3-highgpu-8g` *(8× H100)* | 128 | 512 | 8× H100 | ~$32.00 | Transformer or LLM fine-tuning |
| 1 TB+ | `a4-highgpu-4g` *(4× B200)* | 96 | 768 | 4× B200 | ~$36.00 | Foundation-model scale training |

### Summary

1. Choose the `machine_type` for CPU and memory resources.  
2. Attach a GPU with `accelerator_type` and `accelerator_count` if needed.  
3. Only `A2`, `A3`, and `A4` families include GPUs automatically.  
4. For most research training jobs, `n1-standard-8` + `NVIDIA_TESLA_T4` or `NVIDIA_L4` is a practical and affordable starting point.
