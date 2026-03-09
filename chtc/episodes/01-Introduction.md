---
title: "Overview of CHTC for Machine Learning"
teaching: 10
exercises: 2
---

::::::::::::::::::::::::::::::::::::: questions

- Why would I run ML/AI experiments on CHTC instead of on my laptop or a cloud platform?
- What does CHTC offer for ML/AI, and how is it organized?
- What is the "submit node as controller" pattern?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Identify when CHTC makes sense for ML/AI work.
- Describe what CHTC and HTCondor provide for ML/AI researchers.
- Explain the submit-node-as-controller pattern used throughout this workshop.

::::::::::::::::::::::::::::::::::::::::::::::::

## Why run ML/AI on CHTC?

You have ML/AI code that works on your laptop. But at some point you need more — a bigger GPU, a dataset that won't fit on disk, or the ability to run dozens of training experiments overnight. Commercial cloud platforms let you rent hardware, but the costs add up. **CHTC gives you access to powerful hardware — including A100 and H100 GPUs — completely free.**

### What is CHTC?

The **Center for High Throughput Computing (CHTC)** at UW-Madison provides shared computing resources to the campus research community. CHTC's computing pool is managed by **HTCondor**, a job scheduling system designed for *high throughput computing* (HTC) — running many independent jobs efficiently across a large pool of machines.

**HTC vs HPC:** Traditional High Performance Computing (HPC) focuses on single large jobs that need tightly coupled processors communicating over fast networks (e.g., physics simulations). High Throughput Computing focuses on maximizing the total amount of work completed over time — many independent jobs running in parallel. ML workloads (training runs, hyperparameter sweeps, batch inference) are often embarrassingly parallel and fit the HTC model perfectly.

### CHTC vs. laptop vs. cloud

| Factor | Laptop | CHTC | Cloud (GCP, AWS) |
|--------|--------|------|-------------------|
| **Cost** | Hardware purchase | Free | Pay per hour |
| **GPUs** | Consumer GPU (if any) | A100 40/80 GB, H100 80 GB, H200 141 GB | On-demand (T4, L4, A100, H100) |
| **Parallelism** | 1 job at a time | Hundreds of jobs in parallel | Hundreds of jobs (but each costs $) |
| **Job time limits** | None | 12 hrs (short), 24 hrs (medium), 7 days (long) | None (but meter is running) |
| **Queue wait times** | None | Minutes to hours (varies by demand) | None (on-demand) |
| **Setup** | Install packages locally | SSH + submit files + containers | Cloud console + billing account |
| **Managed services** | None | None — you manage your own workflows | HP tuning, model hosting, MLOps |
| **Internet from workers** | Full | Restricted (workers typically have no internet) | Full |

**The short version:** use CHTC when you need free GPUs, need to run many parallel experiments, or need hardware beyond what your laptop offers. Use cloud when you need managed services (hosted model endpoints, Bayesian HP tuning), multi-node NVLink training, or guaranteed availability with no queue wait.

Many researchers use both — develop and test locally, run experiments on CHTC, then use cloud only for capabilities CHTC doesn't offer.

### When does model size justify CHTC GPUs?

Not every model needs a GPU cluster. Here's a rough guide:

| Model scale | Parameters | Example models | Where to run |
|-------------|-----------|----------------|--------------|
| Small | < 10M | Logistic regression, small CNNs, XGBoost | Laptop — CHTC adds overhead without much benefit |
| Medium | 10M–500M | ResNets, BERT-base, mid-sized transformers | CHTC with a single GPU (or CPU for simpler models) |
| Large | 500M–10B | GPT-2, LLaMA-7B, fine-tuning large transformers | CHTC with A100 (40/80 GB) |
| Very large | 10B–70B | LLaMA-70B, Mixtral | CHTC with H100/H200 (80–141 GB) |
| Frontier | 70B+ | GPT-4-scale, multi-expert models | Cloud — requires multi-node clusters beyond what CHTC offers |

**CHTC's [GPU Lab](https://chtc.cs.wisc.edu/uw-research-computing/gpu-lab) covers more than you might think.** A100s, H100s, and H200s provide enough VRAM to run inference or fine-tune models up to ~70B parameters with quantization. For many UW researchers, this handles "large model" workloads without needing cloud.

### A note on "free" resources

CHTC is free to UW-Madison researchers, but resources are shared. Being a good citizen matters:

- **Don't over-request** resources (CPUs, memory, GPUs) — this takes capacity from other users.
- **Fair-share scheduling** means running many jobs temporarily lowers your priority, so others get a turn.
- **Clean up** old files and output from completed jobs.

We'll cover resource management in detail in [Episode 9](09-Resource-management.md).

## What CHTC provides for ML/AI

CHTC gives you three things that matter for ML/AI research:

**Flexible compute.** You request the hardware your job needs in an HTCondor submit file:

- **CPUs** for lightweight models, preprocessing, or feature engineering.
- **GPUs** (A100, H100, H200) for training deep learning models.
- HTCondor matches your job to a machine with the requested resources.

**Tiered storage.** CHTC provides a storage hierarchy designed for different use cases:

| Storage | Size | Purpose | Persistence |
|---------|------|---------|-------------|
| `/home/<user>/` | 30 GB | Scripts, submit files, small configs | Backed up |
| `/scratch/<user>/` | 100 GB | Intermediate outputs, working data | NOT backed up, auto-purged after 30 days |
| `/staging/<user>/` | 100 GB | Large input/output files for jobs | NOT backed up |

**Containerized environments.** CHTC uses **Apptainer** (formerly Singularity) containers to provide reproducible software environments. You can pull containers from Docker Hub or build your own — similar to how cloud platforms provide prebuilt ML containers.

## How the pieces fit together: the submit-node-as-controller pattern

The central idea of this workshop mirrors the "notebook as controller" pattern from the [GCP version](https://qualiamachine.github.io/Intro_GCP_for_ML/) of this workshop. You work on a lightweight **submit node** — a shared login machine — and use **HTCondor** to dispatch compute-heavy jobs to worker machines in the pool. The submit node itself does not run heavy compute. Instead, it orchestrates:

- **Training jobs** (Eps 4–5) — submit a script that runs on a worker with the CPU/GPU resources you request, then returns results when complete.
- **Hyperparameter sweeps** (Ep 6) — use `queue from params.csv` to launch dozens or hundreds of parallel trials.
- **File transfers** (Ep 3) — HTCondor moves your input files to workers and brings output files back automatically.
- **RAG pipelines** (Ep 7) — run embedding jobs as batch HTCondor work and generation via interactive GPU sessions.

All of these are managed through submit files and HTCondor commands from the submit node. This keeps the submit node lightweight (it's shared with other users) while your actual compute runs on dedicated worker hardware.

| GCP Concept | CHTC Equivalent |
|---|---|
| Vertex AI Workbench notebook (controller) | Submit node (SSH terminal) |
| `aiplatform.CustomTrainingJob` + `job.run()` | HTCondor submit file + `condor_submit` |
| GCS buckets (`gs://`) | `/staging`, `/scratch`, `/home` storage hierarchy |
| Prebuilt containers (pytorch-gpu, xgboost-cpu) | Apptainer `.sif` images (from Docker Hub or custom-built) |
| Vertex AI HP Tuning Job | `queue from params.csv` (embarrassingly parallel) |
| `gcloud` CLI | `condor_submit`, `condor_q`, `condor_status`, `condor_rm` |
| Billing / cost monitoring | Free — but fair-share priority (`condor_userprio`) |

::::::::::::::::::::::::::::::::::::: callout

### Console, notebooks, or CLI?

On GCP, you can use a web console, notebooks, or CLI. On CHTC, **the CLI is the primary interface** — there is no web console for job management. Everything is done via SSH and command-line tools. If you're more comfortable with graphical interfaces, don't worry — the commands are straightforward and this workshop walks you through each one.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Your current setup

Think about how you currently run ML experiments:

- What hardware do you use — laptop, HPC cluster, cloud?
- What's the biggest infrastructure pain point in your workflow (GPU access, environment setup, data transfer, cost)?
- What would you most like to offload to CHTC?

Take 3–5 minutes to discuss with a partner or share in the workshop chat.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- CHTC provides free access to powerful GPUs (A100, H100, H200) and unlimited job parallelism for UW-Madison researchers.
- High Throughput Computing (HTC) is ideal for ML workloads: many independent training runs, hyperparameter sweeps, and batch inference jobs.
- The submit-node-as-controller pattern keeps the submit node lightweight while dispatching heavy compute to HTCondor workers.
- CHTC uses a tiered storage hierarchy (`/home`, `/scratch`, `/staging`) and Apptainer containers for reproducible environments.

::::::::::::::::::::::::::::::::::::::::::::::::
