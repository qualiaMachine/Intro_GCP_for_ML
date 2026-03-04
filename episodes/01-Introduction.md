---
title: "Overview of Google Cloud for Machine Learning and AI"
teaching: 10
exercises: 2
---

::::::::::::::::::::::::::::::::::::: questions

- Why would I run ML/AI experiments in the cloud instead of on my laptop or an HPC cluster?
- What does GCP offer for ML/AI, and how is it organized?
- What is the "notebook as controller" pattern?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Identify when cloud compute makes sense for ML/AI work.
- Describe what GCP and Vertex AI provide for ML/AI researchers.
- Explain the notebook-as-controller pattern used throughout this workshop.

::::::::::::::::::::::::::::::::::::::::::::::::

## Why run ML/AI in the cloud?

You have ML/AI code that works on your laptop. But at some point you need more — a bigger GPU (or multiple GPUs), a dataset that won't fit on disk, or the ability to run dozens of training experiments overnight. You could invest in local hardware or compete for time on a shared HPC cluster, but cloud platforms let you rent exactly the hardware you need, for exactly as long as you need it, and then shut it down.

### Cloud vs. university HPC clusters

Most universities offer shared HPC clusters with GPUs. These are excellent resources — but they have tradeoffs worth understanding:

| Factor | University HPC | Cloud (GCP) |
|--------|---------------|-------------|
| **Cost** | Free or subsidized | Pay per hour |
| **GPU availability** | Shared queue; wait times during peak periods | On-demand (subject to quota) |
| **Hardware variety** | Fixed hardware refresh cycle (3–5 years) | Latest GPUs available immediately (A100, H100, L4) |
| **Scaling** | Limited by cluster size | Spin up hundreds of jobs in parallel |
| **Multi-GPU / NVLink** | Sometimes available, depends on cluster | Available on demand (e.g., A2/A3 instances with NVLink-connected multi-GPU nodes) |
| **Job orchestration** | Writing scheduler scripts, packaging environments, and wiring up parallel job arrays can take days of refactoring | A few SDK calls: define a job, set hardware, call `.run()` — parallelism (e.g., tuning trials) is built in |
| **Software environment** | Module system; may need IT support for new packages | Full root access in containers; install anything |
| **Data governance** | On-campus, known compliance posture | Requires configuring IAM, encryption, region controls |

**The short version:** use your university cluster when it has the hardware you need and the queue isn't blocking you. Use the cloud when you need hardware your cluster doesn't have, need to scale beyond what the queue allows, or need a specific software environment you can't easily get on-campus.

Many researchers use both — develop and test on HPC, then scale to cloud for large experiments or specialized hardware. This workshop teaches the cloud side of that workflow.

### Scalability beyond a single GPU

Cloud platforms give you access to compute that's hard to replicate locally:

- **Multi-GPU nodes.** GCP's A2 and A3 machine families provide 1–16 GPUs per node connected via **NVLink** (high-bandwidth GPU-to-GPU interconnects). This matters for distributed training, where GPUs need to exchange gradients quickly — NVLink provides 600+ GB/s bandwidth compared to ~32 GB/s over PCIe.
- **Elastic scaling with minimal code.** Need to run 100 hyperparameter tuning trials? On Vertex AI you define the search space, set `max_trial_count`, and the platform handles provisioning, parallelism, and teardown. On a university HPC cluster the same task often means writing Slurm/Condor job arrays, packaging environments into containers or tarballs, and wiring up result collection — work that can take days or weeks of refactoring before you run a single trial.
- **Power and cooling are someone else's problem.** A single A100 GPU draws ~400W under load. A rack of 8 draws ~3.2kW just for the GPUs. Cloud providers handle the power infrastructure, cooling, and hardware failures.

We won't use multi-GPU training in this workshop — our datasets are small and the models we train fit comfortably on a single GPU — but understanding what's available helps you plan for larger projects.

### When does model size justify cloud compute?

Not every model needs cloud hardware. Here's a rough guide:

| Model scale | Parameters | Example models | Where to run |
|-------------|-----------|----------------|--------------|
| Small | < 10M | Logistic regression, small CNNs, XGBoost | Laptop or HPC — cloud adds overhead without much benefit |
| Medium | 10M–500M | ResNets, BERT-base, mid-sized transformers | HPC or cloud with a single GPU (T4 or L4) |
| Large | 500M–10B | GPT-2, LLaMA-7B, fine-tuning large transformers | Cloud — requires A100-class GPUs with 40–80 GB VRAM |
| Very large | 10B+ | LLaMA-70B, Gemini, GPT-4-scale models | Cloud — requires multi-GPU nodes (A3/H100 or larger) |

**Large language models (LLMs) are a strong use case for cloud.** Fine-tuning a 7B-parameter model requires ~28 GB of GPU memory just for the model weights (in mixed precision), plus memory for gradients and optimizer states — easily exceeding what a single consumer GPU provides. Inference with these models has similar requirements. Cloud platforms give you on-demand access to A100 (40/80 GB), H100 (80 GB), and B200 GPUs without purchasing hardware that costs tens of thousands of dollars.

For smaller models (under ~500M parameters), a university HPC cluster with T4 or V100 GPUs is often sufficient. Cloud becomes the clear choice when your model or dataset outgrows local hardware, or when you need rapid iteration across many experiments.

Google Cloud Platform (GCP) is one of several clouds that supports this. The rest of this episode explains what GCP offers for ML/AI and how the pieces fit together.

## What GCP provides for ML/AI

GCP gives you three things that matter for applied ML/AI research:

**Flexible compute.** You pick the hardware that fits your workload:

- **CPUs** for lightweight models, preprocessing, or feature engineering.
- **GPUs** (NVIDIA T4, L4, V100, A100, H100) for training deep learning models. For help choosing, see [Instances for ML on GCP](../instances-for-ML.html).
- **TPUs** (Tensor Processing Units) — Google's custom hardware for matrix-heavy workloads. TPUs work best with TensorFlow and JAX; PyTorch support is improving but still less mature.

**Scalable storage.** Google Cloud Storage (GCS) buckets give you a place to store datasets, scripts, and model artifacts that any job or notebook can access. Think of it as a shared filesystem for your project.

**Managed ML/AI services.** Vertex AI is Google's ML/AI platform. It wraps compute, storage, and tooling into a set of services designed for ML/AI workflows — managed notebooks, training jobs, hyperparameter tuning, model hosting, and access to foundation models like Gemini.

## How the pieces fit together: Vertex AI

Google Cloud has many products and brand names. Here are the ones you'll use in this workshop and how they relate:

| Term | What it is |
|------|-----------|
| **GCP** | Google Cloud Platform — the overall cloud: compute, storage, networking. |
| **Vertex AI** | Google's ML platform — notebooks, training jobs, tuning, model hosting. Everything below lives under this umbrella. |
| **Workbench** | Managed Jupyter notebooks that run on a Compute Engine VM. Your interactive environment. |
| **Training & tuning jobs** | How you run code on Vertex AI hardware. You submit a script and a machine spec; Vertex AI provisions the VM, runs it, and shuts it down. The SDK offers several flavors — `CustomTrainingJob` (Ep 4–5), `HyperparameterTuningJob` (Ep 6) — and the CLI equivalent is `gcloud ai custom-jobs` (Ep 8). |
| **Cloud Storage (GCS)** | Object storage for files. Similar to AWS S3. |
| **Compute Engine** | Virtual machines you configure with CPUs, GPUs, or TPUs. Workbench and training jobs run on Compute Engine under the hood. |
| **Gemini** | Google's family of large language models, accessed through the Vertex AI API. |

For a full list of terms, see the [Glossary](../learners/reference.md).

## The notebook-as-controller pattern

The central idea of this workshop is simple. You work in a lightweight **Vertex AI Workbench** notebook — a small, cheap VM where you explore data, visualize results, and iterate. When it's time to do heavy lifting, you use the **Vertex AI Python SDK** to launch jobs on separate, powerful machines:

- **Training jobs** run your training script on auto-provisioned GPU hardware, then shut down when complete.
- **Hyperparameter tuning jobs** search a parameter space across parallel trials and return the best configuration and metrics.

Both job types write their outputs — model artifacts, checkpoints, logs, metrics, and trial results — to **Google Cloud Storage (GCS)**, which acts as shared persistent storage. You then pull results back into your notebook to inspect and iterate.

This keeps costs low (the notebook VM stays small) and keeps your work reproducible (each job is a clean, logged run on dedicated hardware).

Here's how the pieces connect:

![Notebook as a controller](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/diagram1_training_and_tuning.svg){alt="Notebook as a controller diagram"}

The same notebook can also orchestrate **Retrieval-Augmented Generation (RAG)** pipelines — chunking documents, building embeddings via the Gemini API, and generating grounded answers, all with data stored in GCS:

![RAG pipeline with Gemini API](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/diagram2_rag_gemini.svg){alt="Architecture diagram showing the RAG pipeline: a Workbench notebook orchestrates document chunking, embedding via the Gemini API, and retrieval-augmented generation, with documents and embeddings stored in a GCS bucket."}

::::::::::::::::::::::::::::::::::::: callout

### Console, notebooks, or CLI — your choice

This workshop uses the **GCP web console** and **Workbench notebooks** for most tasks because they're visual and easy to follow for beginners. But nearly everything we do can also be done from the **`gcloud` command-line tool** — submitting training jobs, managing buckets, checking quotas. [Episode 8](08-CLI-workflows.md) covers the CLI equivalents. If you prefer terminal-based workflows or need to automate jobs in scripts and CI/CD pipelines, that episode shows you how.

**One important caveat:** whether you use the console, notebooks, or CLI, resources you create (VMs, training jobs, endpoints) keep running and billing until you explicitly stop them. There's no automatic shutdown. We cover cleanup habits in [Episode 9](09-Resource-management-cleanup.md), but the short version is: always check for running resources before you walk away.

::::::::::::::::::::::::::::::::::::::::::::::::

## A note on cloud costs

Cloud computing is not free, but it's worth putting costs in context:

- **Hardware is expensive and ages fast.** A single A100 GPU costs ~$15,000 and is outdated within a few years. Cloud lets you rent the latest hardware by the hour.
- **You pay only for what you use.** Stop a VM and the meter stops — valuable for bursty research workloads.
- **Budgets and alerts keep you safe.** GCP billing dashboards and budget alerts help prevent surprise bills. We cover cleanup in [Episode 9](09-Resource-management-cleanup.md).

The key habit: choose the right machine size, stop resources when idle, and monitor spending. We'll reinforce this throughout.

::::::::::::::::::::::::::::::::::::: callout

### For UW-Madison researchers

UW-Madison offers reduced-overhead cloud billing, NIH STRIDES discounts, Google Cloud research credits (up to $5,000), free on-campus GPUs via [CHTC](https://chtc.cs.wisc.edu/), and dedicated support from the [Public Cloud Team](mailto:cloud-services@cio.wisc.edu). See the [UW-Madison Cloud Resources](../uw-madison-cloud-resources.html) page for details.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Your current setup

Think about how you currently run ML experiments:

- What hardware do you use — laptop, HPC cluster, cloud?
- What's the biggest infrastructure pain point in your workflow (GPU access, environment setup, data transfer, cost)?
- What would you most like to offload to a managed service?

Take 3–5 minutes to discuss with a partner or share in the workshop chat.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Cloud platforms let you rent hardware on demand instead of buying or waiting for shared resources.
- GCP organizes its ML/AI services under Vertex AI — notebooks, training jobs, tuning, and model hosting.
- The notebook-as-controller pattern keeps your notebook cheap while offloading heavy training to dedicated Vertex AI jobs.
- Everything in this workshop can also be done from the `gcloud` CLI ([Episode 8](08-CLI-workflows.md)).

::::::::::::::::::::::::::::::::::::::::::::::::
