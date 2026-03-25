---
title: "Overview of CHTC for Machine Learning and AI"
teaching: 10
exercises: 2
---

::::::::::::::::::::::::::::::::::::: questions

- Why would I run ML/AI experiments on CHTC instead of on my laptop?
- What does CHTC offer for ML/AI, and how is it organized?
- What is the "submit node as controller" pattern?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Identify when high-throughput computing makes sense for ML/AI work.
- Describe what CHTC and HTCondor provide for ML/AI researchers.
- Explain the submit-node pattern used throughout this workshop.

::::::::::::::::::::::::::::::::::::::::::::::::

## Why run ML/AI on CHTC?

You have ML/AI code that works on your laptop. But at some point you need more — a bigger GPU (or multiple GPUs), a dataset that won't fit in memory, or the ability to run dozens of training experiments overnight. You could invest in local hardware, but UW-Madison's **Center for High Throughput Computing (CHTC)** lets you access powerful shared hardware — including cutting-edge GPUs — for free, on demand.

### What is CHTC?

[CHTC](https://chtc.cs.wisc.edu/) is a research computing center at UW-Madison that provides large-scale computing resources to the campus community. It uses **HTCondor**, a job scheduling system developed at UW-Madison, to manage and distribute computational work across a large pool of shared machines.

Key features for ML/AI researchers:

- **Free for UW-Madison researchers** — no billing, no credits to manage, no surprise charges.
- **GPU Lab** with NVIDIA A100 (40/80 GB), H100 (80 GB), and H200 (141 GB) GPUs — enough to fine-tune models up to ~70B parameters with quantization.
- **Docker container support** — bring your own software environment via Docker images.
- **Massive throughput** — run hundreds of independent jobs in parallel (e.g., hyperparameter sweeps).
- **Dedicated support** — CHTC's facilitation team helps researchers optimize their workflows.

### Laptop vs. CHTC

| Factor | Laptop | CHTC |
|--------|--------|------|
| **Cost** | Your own hardware | Free for UW researchers |
| **GPU availability** | Whatever you bought | A100, H100, H200 — shared queue |
| **Scaling** | One machine | Hundreds of jobs in parallel |
| **Software environment** | Manage yourself | Docker containers, reproducible |
| **Job runtime** | Limited by your patience | 12 hrs (short), 24 hrs (medium), 7 days (long) |
| **Storage** | Local disk | Home, staging, and SQUID filesystems |

**The short version:** use your laptop for development and quick tests. Use CHTC when you need more hardware, more parallelism, or longer runtimes than your laptop can provide.

### When does model size justify CHTC?

Not every model needs CHTC. Here's a rough guide:

| Model scale | Parameters | Example models | Where to run |
|-------------|-----------|----------------|--------------|
| Small | < 10M | Logistic regression, small CNNs, XGBoost | Laptop — CHTC adds overhead without much benefit |
| Medium | 10M–500M | ResNets, BERT-base, mid-sized transformers | CHTC with a single GPU (T4, L40, A100) |
| Large | 500M–10B | GPT-2, LLaMA-7B, fine-tuning large transformers | CHTC GPU Lab with A100 (40/80 GB) |
| Very large | 10B–70B | LLaMA-70B, Mixtral | CHTC GPU Lab with H100/H200 (80–141 GB) |
| Frontier | 70B+ | GPT-4-scale, multi-expert models | Cloud platforms — requires multi-node clusters beyond what most HTC queues offer |

**CHTC's [GPU Lab](https://chtc.cs.wisc.edu/uw-research-computing/gpu-lab) covers more than you might think.** It includes A100s (40 and 80 GB), H100s (80 GB), and H200s (141 GB) — enough VRAM to run inference or fine-tune models up to ~70B parameters on a single GPU with quantization. For many UW researchers, this hardware handles "large model" workloads without needing cloud.

Cloud becomes the clear choice when you need interconnected multi-GPU nodes (NVLink) for large distributed training, or hardware beyond what the GPU Lab queue offers.

### A note on cost (or lack thereof)

Unlike cloud platforms, **CHTC is free for UW-Madison researchers**. There are no per-hour charges, no billing accounts, and no surprise invoices. The "cost" is shared resources: be a good citizen, request only what you need, and clean up after yourself. We'll cover resource etiquette in [Episode 9](09-Resource-management-best-practices.md).

## What CHTC provides for ML/AI

CHTC gives you three things that matter for applied ML/AI research:

**Flexible compute.** You request the hardware that fits your workload:

- **CPUs** for lightweight models, preprocessing, or feature engineering.
- **GPUs** (NVIDIA A100, H100, H200, L40, T4) for training deep learning models. For help choosing, see [Compute for ML](../compute-for-ML.html).

**Scalable storage.** CHTC provides multiple storage tiers for different use cases:

- `/home` — small files, submit scripts, code (~20 GB quota).
- `/staging` — larger datasets and outputs transferred to/from jobs.
- **SQUID** — large, read-only datasets shared across many jobs via HTTP.

**Containerized environments.** HTCondor runs your jobs inside Docker containers, so you get a fully reproducible software environment (PyTorch, XGBoost, TensorFlow, etc.) without installing anything on the shared machines.

## How the pieces fit together: HTCondor

Here are the key components you'll use in this workshop:

| Term | What it is |
|------|-----------|
| **CHTC** | Center for High Throughput Computing — UW-Madison's research computing center. |
| **HTCondor** | The job scheduling system that manages compute resources. You submit jobs; HTCondor finds machines to run them. |
| **Submit node** | The server you SSH into. This is where you write code, prepare data, and submit jobs. It is *not* for heavy computation. |
| **Execute node** | The machine where your job actually runs. HTCondor assigns one automatically based on your resource request. |
| **Submit file** (`.sub`) | A configuration file that tells HTCondor what to run, what resources you need, and where to find your data. |
| **Docker container** | A packaged software environment. You specify a Docker image in your submit file, and HTCondor runs your code inside it. |
| **DAGMan** | HTCondor's workflow manager for multi-step or dependent jobs (used in [Episode 8](08-Advanced-HTCondor-workflows.md)). |

For a full list of terms, see the [Glossary](../learners/reference.md).

## The submit-node pattern

The central idea of this workshop is simple: you work on a **submit node** — a shared server — and use **HTCondor submit files** to dispatch work to execute nodes. The submit node itself does not run heavy compute. Instead, it orchestrates:

- **Training jobs** (Eps 4–5) — run your script on a machine with the hardware you requested, then release resources when complete.
- **Hyperparameter tuning** (Ep 6) — search a parameter space across parallel jobs and collect results.
- **Data staging** (Ep 3) — prepare and transfer data between storage tiers and job execution.
- **RAG pipelines** (Ep 7) — run embedding and generation workloads as HTCondor jobs.

All of these are managed through submit files and command-line tools. This keeps the submit node free for other users and ensures your work is reproducible (each job is a clean, logged run on dedicated hardware).

::::::::::::::::::::::::::::::::::::: callout

### Terminal-based workflow

Unlike cloud platforms with web consoles, CHTC workflows are entirely **terminal-based**. You'll SSH into a submit node, write submit files in a text editor, and manage jobs with commands like `condor_submit`, `condor_q`, and `condor_status`. If you're comfortable with the command line, you'll feel right at home. If you're new to it, don't worry — we'll walk through every step.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Your current setup

Think about how you currently run ML experiments:

- What hardware do you use — laptop, HPC cluster, cloud?
- What's the biggest infrastructure pain point in your workflow (GPU access, environment setup, data transfer, runtime limits)?
- What would you most like to offload to shared compute?

Take 3–5 minutes to discuss with a partner or share in the workshop chat.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- CHTC provides free, shared compute resources for UW-Madison researchers — no billing required.
- HTCondor schedules your jobs onto available hardware, including GPUs, automatically.
- The submit-node pattern keeps the shared login server free while your jobs run on dedicated execute nodes.
- CHTC's GPU Lab includes A100, H100, and H200 GPUs — sufficient for most research ML/AI workloads.
- Everything in this workshop uses terminal-based workflows with HTCondor submit files.

::::::::::::::::::::::::::::::::::::::::::::::::
