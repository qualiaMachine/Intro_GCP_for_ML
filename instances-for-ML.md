---
title: Instances for ML 
---

The below table provides general recommendations for selecting Google Cloud Platform (GCP) instances. 

**All pricing estimates** are based on public rates for notebooks running in us-central1 as of October 2025.  
Actual cost depends on sustained-use discounts and attached GPU storage quotas. See the full pricing tables for the most up-to-date info:  

**Reference Docs:**
- [Compute Engine VM Instance Pricing (applies to notebook backends)](https://cloud.google.com/compute/vm-instance-pricing)
- [Compute Engine GPU Pricing](https://cloud.google.com/compute/gpus-pricing)
- [All Compute Pricing Overview](https://cloud.google.com/compute/all-pricing)
  
#### General Notes:
- **Minimum RAM** should be at least 1.5× dataset size unless using batch processing (common in deep learning).  
- The **N2** and **C2** series are optimized for CPU-heavy tasks such as preprocessing, feature engineering, and traditional ML.  
- **GPU choices** depend on task size and budget — T4 (cost-effective), A100 (high-performance), H100/B200 (cutting-edge).  
- The **A2** family (A100 GPUs) offers the best cost/performance trade-off for most research-scale DL workloads.  
- The **A3/A4** families target large-scale model training and are overkill for most notebook-level experiments.  
- **Free Tier Eligibility**: Some smaller instance types (e.g., `e2-micro`) may be eligible for the [GCP Free Tier](https://cloud.google.com/free). Check usage limits before running persistent notebooks.


| **Dataset Size** | **Recommended Notebook Instance (GCP)** | **vCPU** | **Memory (GiB)** | **GPU** | **Price per Hour (USD)** | **Suitable Tasks** |
|------------------|------------------------------------------|----------|------------------|---------|--------------------------|--------------------|
| < 1 GB           | `e2-micro` *(Free Tier)*                | 2        | 1                | None    | Free Tier Eligible       | Simple scripts, test notebooks, lightweight tasks |
| < 1 GB           | `n2-standard-4` *(Workbench)*           | 4        | 16               | None    | ~$0.17                   | Preprocessing, regression, small models |
| < 1 GB           | `a2-highgpu-1g` *(1× A100)*             | 12       | 85               | 1× NVIDIA A100 | ~$2.93 | Entry-level GPU experiments, fine-tuning small DL models |
| 10 GB            | `c2-standard-8` *(Workbench)*           | 8        | 32               | None    | ~$0.34                   | CPU-heavy model training, feature engineering |
| 10 GB            | `n2-standard-8` *(Workbench)*           | 8        | 32               | None    | ~$0.38                   | Preprocessing, small- to mid-scale ML |
| 10 GB            | `a2-highgpu-2g` *(2× A100)*             | 24       | 170              | 2× NVIDIA A100 | ~$5.90 | Moderate deep learning workloads |
| 50 GB            | `c2-standard-16` *(Workbench)*          | 16       | 64               | None    | ~$0.68                   | Large CPU training tasks, data prep |
| 50 GB            | `n2-standard-16` *(Workbench)*          | 16       | 64               | None    | ~$0.70                   | Feature engineering, large non-GPU models |
| 50 GB            | `a2-highgpu-4g` *(4× A100)*             | 48       | 340              | 4× NVIDIA A100 | ~$11.80 | Mid-scale deep learning, cost-balanced training |
| 100 GB           | `a3-highgpu-8g` *(8× H100)*             | 128      | 512              | 8× NVIDIA H100 | ~$32.00 | Transformer training, distributed DL experiments |
| 100 GB           | `a4-highgpu-4g` *(4× B200)*             | 96       | 768              | 4× NVIDIA B200 | ~$36.00 | Large-scale DL with advanced GPUs |
| 1 TB+            | `a4x-highgpu-4g` *(4× B200)*            | 144      | 884              | 4× NVIDIA B200 | ~$40.00 | Foundation model research, large-batch DL |
| 1 TB+            | `a3-mega` *(8× H100)*                   | 192      | 1024             | 8× NVIDIA H100 | ~$45.00 | Distributed large-model training across multiple GPUs |


