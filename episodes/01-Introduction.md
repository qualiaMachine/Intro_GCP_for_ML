---
title: "Overview of Google Cloud for Machine Learning"
teaching: 10
exercises: 2
---

::::::::::::::::::::::::::::::::::::: questions

- Why would I run ML experiments in the cloud instead of on my laptop or an HPC cluster?
- What does GCP offer for machine learning, and how is it organized?
- What is the "notebook as controller" pattern?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Identify when cloud compute makes sense for ML work.
- Describe what GCP and Vertex AI provide for ML researchers.
- Explain the notebook-as-controller pattern used throughout this workshop.

::::::::::::::::::::::::::::::::::::::::::::::::

## Why run ML in the cloud?

You have ML code that works on your laptop. But at some point you need more — a bigger GPU, a dataset that won't fit on disk, or the ability to run dozens of training experiments overnight. You could invest in local hardware or compete for time on a shared HPC cluster, but cloud platforms let you rent exactly the hardware you need, for exactly as long as you need it, and then shut it down.

Google Cloud Platform (GCP) is one of several clouds that supports this. The rest of this episode explains what GCP offers for ML and how the pieces fit together.

## What GCP provides for ML

GCP gives you three things that matter for applied ML research:

**Flexible compute.** You pick the hardware that fits your workload:

- **CPUs** for lightweight models, preprocessing, or feature engineering.
- **GPUs** (NVIDIA T4, V100, A100, etc.) for training deep learning models.
- **TPUs** (Tensor Processing Units) — Google's custom hardware for matrix-heavy workloads. TPUs work best with TensorFlow and JAX; PyTorch support is improving but still less mature.

**Scalable storage.** Google Cloud Storage (GCS) buckets give you a place to store datasets, scripts, and model artifacts that any job or notebook can access. Think of it as a shared filesystem for your project.

**Managed ML services.** Vertex AI is Google's ML platform. It wraps compute, storage, and tooling into a set of services designed for ML workflows — managed notebooks, training jobs, hyperparameter tuning, model hosting, and access to foundation models like Gemini.

## How the pieces fit together: Vertex AI

Google Cloud has many products and brand names. Here are the ones you'll use in this workshop and how they relate:

| Term | What it is |
|------|-----------|
| **GCP** | Google Cloud Platform — the overall cloud: compute, storage, networking. |
| **Vertex AI** | Google's ML platform — notebooks, training jobs, tuning, model hosting. Everything below lives under this umbrella. |
| **Workbench** | Managed Jupyter notebooks that run on a Compute Engine VM. Your interactive environment. |
| **Custom Jobs** | Managed training runs on dedicated hardware. You submit code; Vertex AI provisions a machine, runs it, and shuts it down. |
| **Cloud Storage (GCS)** | Object storage for files. Similar to AWS S3. |
| **Compute Engine** | Virtual machines you configure with CPUs, GPUs, or TPUs. Workbench and training jobs run on Compute Engine under the hood. |
| **Gemini** | Google's family of large language models, accessed through the Vertex AI API. |

For a full list of terms, see the [Glossary](../learners/glossary.md).

## The notebook-as-controller pattern

The central idea of this workshop is simple. You work in a small, cheap Jupyter notebook in the cloud (a **Vertex AI Workbench** instance). That notebook is your control panel — you write code, explore data, and inspect results there. But when it's time to train a model, you don't train inside the notebook. Instead, you use the **Vertex AI Python SDK** to submit a training job to a separate, more powerful machine (with GPUs, more memory, etc.). When the job finishes, results land in **Cloud Storage** and you pull them back into your notebook.

This keeps costs low (the notebook VM is small) and keeps your work reproducible (each job is a clean, logged run on dedicated hardware).

::::::::::::::::::::::::::::::::::::: callout

### Console, notebooks, or CLI — your choice

This workshop uses the **GCP web console** and **Workbench notebooks** for most tasks because they're visual and easy to follow. But nearly everything we do can also be done from the **`gcloud` command-line tool** — submitting training jobs, managing buckets, checking quotas. [Episode 11](11-CLI-workflows.md) covers the CLI equivalents. If you prefer terminal-based workflows or need to automate jobs in scripts and CI/CD pipelines, that episode shows you how.

::::::::::::::::::::::::::::::::::::::::::::::::

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
- GCP organizes its ML services under Vertex AI — notebooks, training jobs, tuning, and model hosting.
- The notebook-as-controller pattern keeps your notebook cheap while offloading heavy training to dedicated Vertex AI jobs.
- Everything in this workshop can also be done from the `gcloud` CLI ([Episode 11](11-CLI-workflows.md)).

::::::::::::::::::::::::::::::::::::::::::::::::
