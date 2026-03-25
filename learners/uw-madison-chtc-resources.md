---
title: UW-Madison CHTC Resources
---

This page collects UW-Madison-specific CHTC and research computing resources, contacts, and related services relevant to ML/AI researchers. It is meant as a companion to the workshop material and a starting point for learners who want to continue using CHTC after the workshop.

## About CHTC

The [Center for High Throughput Computing (CHTC)](https://chtc.cs.wisc.edu/) is a research computing center at UW-Madison that provides large-scale computing resources to the campus community at **no cost** to UW researchers. CHTC is home to the [HTCondor project](https://htcondor.org/), the workload management system used to schedule and manage jobs.

### Key resources

- **High Throughput Computing (HTC)** — Thousands of CPU cores for running many independent jobs in parallel. Ideal for hyperparameter sweeps, data preprocessing, and embarrassingly parallel workloads.
- **GPU Lab** — Hundreds of GPUs including NVIDIA A100 (40/80 GB), H100 (80 GB), H200 (141 GB), L40, and T4. Supports ML model training, fine-tuning, and inference.
- **High Performance Computing (HPC)** — A dedicated cluster for tightly coupled parallel workloads (MPI, multi-node training). Contact CHTC if your workload requires this.
- **Large-scale data staging** — `/staging` and SQUID provide storage for datasets too large for `/home`.

### How to get an account

1. Visit the [CHTC Account Request Form](https://chtc.cs.wisc.edu/uw-research-computing/form).
2. Fill out the form describing your research and computing needs.
3. CHTC staff will review your request and set up your account (typically within a few business days).

### Getting help

CHTC offers multiple support channels:

- **Email**: [chtc@cs.wisc.edu](mailto:chtc@cs.wisc.edu) — the best way to get help with specific issues.
- **Office hours**: CHTC holds weekly drop-in office hours. Check the [CHTC website](https://chtc.cs.wisc.edu/) for the current schedule.
- **Research facilitation**: CHTC facilitators can help you design your workflow, optimize job submissions, and troubleshoot issues. Request a consultation via email.
- **Documentation**: [CHTC Guides](https://chtc.cs.wisc.edu/uw-research-computing/guides) cover everything from getting started to advanced workflows.

## CHTC vs. cloud computing

CHTC and cloud platforms (AWS, GCP, Azure) serve different needs. Here's a quick comparison:

| Factor | CHTC | Cloud (GCP/AWS/Azure) |
|--------|------|----------------------|
| **Cost** | Free for UW researchers | Pay per hour |
| **GPU access** | Shared queue; wait times during peak periods | On-demand (subject to quota) |
| **Hardware variety** | A100, H100, H200, L40, T4 | Latest GPUs immediately available |
| **Scaling** | Hundreds of parallel jobs | Essentially unlimited |
| **Multi-GPU / NVLink** | Limited multi-GPU support | Available on demand |
| **Software environment** | Docker containers | Managed containers + cloud services |
| **Managed services** | None (you run your own code) | Managed training, tuning, deployment |
| **Data storage** | /home, /staging, SQUID | Cloud storage (S3, GCS) |
| **Runtime limits** | 12 hrs / 24 hrs / 7 days | No limits (pay for what you use) |

**Recommendation:** Start with CHTC — it's free and covers most research ML/AI workloads. Move to cloud when you need managed services, unlimited scaling, or hardware beyond what CHTC offers.

## Other on-campus compute options

### BadgerCompute

[BadgerCompute](https://uw-madison-datascience.github.io/ML-X-Nexus/Toolbox/Compute/BadgerCompute.html) is a lightweight, NetID-authenticated Jupyter notebook service available to UW-Madison users. It is suitable for quick prototyping and small-scale work.

### Google Colab

[Google Colab](https://uw-madison-datascience.github.io/ML-X-Nexus/Toolbox/Compute/GoogleColab.html) provides free cloud-based Jupyter notebooks with optional GPU access. Useful for quick experiments and teaching.

### Cloud platforms

If you need cloud computing, UW-Madison has institutional contracts with AWS, GCP, and Azure that provide negotiated pricing and reduced overhead on grants. See the [UW Cloud Services page](https://uw-madison-datascience.github.io/ML-X-Nexus/Toolbox/Compute/UW-Cloud-Services.html) for details.

## Community and training

- **ML+X Community** — Join [ML+X](https://uw-madison-datascience.github.io/ML-X-Nexus/) for monthly meetings on machine learning and AI at UW-Madison. Contact [endemann@wisc.edu](mailto:endemann@wisc.edu) or join the `#ml-community` channel in the [Data Science Hub Slack](https://hub.datascience.wisc.edu/).
- **CHTC Training** — CHTC periodically offers workshops on HTCondor, GPU computing, and research computing workflows. Check the [CHTC website](https://chtc.cs.wisc.edu/) for upcoming events.
- **RCI** — The [Research Cyberinfrastructure](https://it.wisc.edu/about/division-of-information-technology/research-cyberinfrastructure/) team can help with architecture design and comparing compute options. Email [rci@g-groups.wisc.edu](mailto:rci@g-groups.wisc.edu).

## Related resources

- [Intro to GCP for ML & AI](https://uw-madison-datascience.github.io/ML-X-Nexus/Learn/Workshops/Intro-GCP.html) — Hands-on workshop covering Vertex AI, model training/tuning, and RAG on Google Cloud.
- [Intro to AWS SageMaker for Predictive ML/AI](https://uw-madison-datascience.github.io/ML-X-Nexus/Learn/Workshops/Intro-Amazon_SageMaker.html) — Workshop covering ML workflows in AWS SageMaker.
- [CHTC on Nexus](https://uw-madison-datascience.github.io/ML-X-Nexus/Toolbox/Compute/CHTC.html) — Overview of CHTC resources and how to get started.
