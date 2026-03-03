---
title: "Overview of Google Cloud for Machine Learning"
teaching: 10
exercises: 2
---

::::::::::::::::::::::::::::::::::::: questions

- Why would I run ML experiments in the cloud instead of on my laptop or an HPC cluster?
- What is the "notebook as controller" pattern, and why does this workshop use it?
- What will we actually build during this workshop?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Identify when cloud compute makes sense for ML work.
- Describe the notebook-as-controller pattern used throughout this workshop.
- Map the workshop episodes to the stages of an ML workflow.

::::::::::::::::::::::::::::::::::::::::::::::::

## The problem

You have ML code that works on your laptop. But at some point you need more — a bigger GPU, a dataset that won't fit on disk, or the ability to run dozens of training experiments overnight. You could invest in local hardware or compete for time on a shared HPC cluster, but cloud platforms let you rent exactly the hardware you need, for exactly as long as you need it, and then shut it down.

This workshop teaches you to do that on **Google Cloud Platform (GCP)**, using a pattern we call **notebook as controller**.

## The pattern: notebook as controller

The central idea is simple. You work in a small, cheap Jupyter notebook in the cloud (a **Vertex AI Workbench** instance). That notebook is your control panel — you write code, explore data, and inspect results there. But when it's time to train a model, you don't train inside the notebook. Instead, you use the **Vertex AI Python SDK** to submit a training job to a separate, more powerful machine (with GPUs, more memory, etc.). When the job finishes, results land in **Cloud Storage** and you pull them back into your notebook.

This keeps costs low (the notebook VM is small) and keeps your work reproducible (each job is a clean, logged run on dedicated hardware).

## What we'll build

Over the course of this workshop you'll go from an empty GCP project to a working ML pipeline. Here's the path:

| Episode | What you'll do |
|---------|---------------|
| **2 – Data Storage** | Create a Cloud Storage bucket and upload training data. |
| **3 – Notebooks as Controllers** | Launch a Workbench notebook and connect it to your project. |
| **4 – Accessing Data** | Read data from Cloud Storage into your notebook. |
| **5 – Code Repos** | Pull training scripts from a Git repository into your environment. |
| **6 – Training (CPU)** | Submit your first Vertex AI training job on a CPU machine. |
| **7 – Training (GPU)** | Re-run the same job on a GPU and compare performance. |
| **8 – Hyperparameter Tuning** | Let Vertex AI search for the best model configuration automatically. |
| **9 – Cleanup** | Tear down resources and avoid surprise bills. |
| **10 – RAG** | Build a retrieval-augmented generation pipeline with Gemini. |
| **11 – CLI Workflows** | Run GCP operations from the command line instead of the console. |

Every episode after this one is hands-on. By the end you'll understand not just *how* to use these services, but *when* each one is worth the cost.

## GCP vocabulary cheat sheet

Google Cloud has a lot of brand names. Here are the ones that matter for this workshop:

| Term | What it is |
|------|-----------|
| **GCP** | Google Cloud Platform — the overall cloud: compute, storage, networking. |
| **Vertex AI** | Google's ML platform — notebooks, training jobs, tuning, model hosting. |
| **Workbench** | Managed Jupyter notebooks that run on a Compute Engine VM. |
| **Cloud Storage (GCS)** | Object storage for files (datasets, scripts, model artifacts). Like AWS S3. |
| **Compute Engine** | Virtual machines you configure with CPUs, GPUs, or TPUs. |
| **Gemini** | Google's family of large language models, accessed through Vertex AI. |

For a full list of terms, see the [Glossary](../learners/glossary.md).

## A note on cloud costs

Cloud computing is not free, but it's worth putting costs in context:

- **Hardware is expensive and ages fast.** A single A100 GPU costs ~$15,000 and is outdated within a few years. Cloud lets you rent the latest hardware by the hour.
- **You pay only for what you use.** Stop a VM and the meter stops — valuable for bursty research workloads.
- **Budgets and alerts keep you safe.** GCP billing dashboards and budget alerts help prevent surprise bills. We cover cleanup in [Episode 9](09-Resource-management-cleanup.md).

The key habit: choose the right machine size, stop resources when idle, and monitor spending. We'll reinforce this throughout.

::::::::::::::::::::::::::::::::::::: callout

### For UW-Madison researchers

If you're at UW-Madison, the university offers several resources that can significantly reduce your cloud costs:

- **Reduced overhead on grants** — The [Cloud Computing Pilot](https://rsp.wisc.edu/proposalprep/cloudComputeInfo.cfm) drops F&A overhead from 55.5% to **26%** on cloud expenses when using a UW-provisioned account, saving ~$2,950 per $10,000 spent.
- **NIH STRIDES discounts** — Negotiated pricing on GCP, AWS, and Azure services for NIH-funded researchers. See [STRIDES at UW-Madison](https://kb.wisc.edu/109813) or contact [STRIDES@nih.gov](mailto:STRIDES@nih.gov).
- **Google Cloud Research Credits** — Up to **$5,000** in free credits for faculty and postdocs ($1,000 for PhD students). [Apply here](https://edu.google.com/intl/ALL_us/programs/credits/research/).
- **Free on-campus GPUs** — [CHTC](https://chtc.cs.wisc.edu/) provides access to hundreds of GPUs (including A100s) at no cost.
- **Support** — The [Public Cloud Team](mailto:cloud-services@cio.wisc.edu) offers weekly office hours and architecture consultations. The [ML+X community](https://hub.datascience.wisc.edu/communities/mlx/) holds monthly meetings on ML/AI topics.

See the [UW-Madison Cloud Resources](../uw-madison-cloud-resources.html) page for the full list.

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
- The notebook-as-controller pattern keeps your notebook cheap while offloading heavy training to dedicated Vertex AI jobs.
- This workshop walks through the full cycle: data storage → notebook setup → training → tuning → cleanup.

::::::::::::::::::::::::::::::::::::::::::::::::
