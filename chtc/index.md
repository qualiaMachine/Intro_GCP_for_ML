---
site: sandpaper::sandpaper_site
---

Already know how to train an ML model in Python but want to scale up for free? This hands-on workshop gets you running ML/AI workloads on **UW-Madison's Center for High Throughput Computing (CHTC)** — no prior HTC experience required. By the end, you'll be able to move a local training workflow onto CHTC's **HTCondor** pool and take advantage of free GPUs, massive parallelism, and containerized environments.

**What you'll learn:**

- **Submit node workflows** — Use an SSH terminal as your controller to dispatch compute jobs across the CHTC pool.
- **Data management** — Transfer datasets to and from HTCondor jobs using CHTC's storage hierarchy (`/home`, `/scratch`, `/staging`).
- **Scalable model training** — Launch XGBoost (CPU) and PyTorch (GPU) training jobs using Apptainer containers on CHTC hardware including A100 GPUs.
- **Hyperparameter tuning** — Run massively parallel parameter sweeps using HTCondor's `queue from` syntax — unlimited trials, zero cost.
- **RAG pipelines** — Build a retrieval-augmented generation pipeline using open-source models (sentence-transformers + local LLM) on CHTC GPUs.
- **Resource management** — Understand fair-share priority, diagnose held jobs, and practice good resource citizenship.

#### Prerequisites

This workshop assumes you have a **fundamental ML/AI background**. Specifically, you should be comfortable with:

- **Python** — writing scripts, using packages like pandas and NumPy. New to Python? See the [Intro to Python](https://uw-madison-datascience.github.io/ML-X-Nexus/Learn/Workshops/Intro-Python_Gapminder.html) workshop.
- **Core ML/AI concepts** — train/test splits, overfitting, loss functions, hyperparameters. New to ML/AI? See the [Intro to Machine Learning](https://uw-madison-datascience.github.io/ML-X-Nexus/Learn/Workshops/Intro-ML_Sklearn.html) workshop.
- **Training a model** — you've trained at least one model in any framework (scikit-learn, PyTorch, TensorFlow, XGBoost, etc.).
- **Command line basics** — navigating directories, running commands in a terminal. You should be comfortable with SSH.

No prior CHTC or HTCondor experience is required — that's what this workshop teaches.
