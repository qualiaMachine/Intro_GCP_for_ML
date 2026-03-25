---
site: sandpaper::sandpaper_site
---

Already know how to train an ML model in Python but haven't used high-throughput computing? This hands-on workshop gets you running ML/AI workloads on the **Center for High Throughput Computing (CHTC)** at UW-Madison — no prior HTC experience required. By the end, you'll be able to move a local training workflow onto CHTC's **HTCondor** system and take advantage of shared GPUs, scalable job submission, and containerized environments.

**What you'll learn:**

- **Connecting to CHTC** — Log in to a submit node and navigate the CHTC filesystem.
- **Data management** — Stage datasets for HTCondor jobs using `/home`, `/staging`, and SQUID.
- **Scalable model training** — Submit HTCondor jobs that run your PyTorch (or other framework) code on CPUs or GPUs.
- **Hyperparameter tuning** — Use HTCondor's `queue` mechanisms and DAGMan to run parallel tuning sweeps.
- **RAG pipelines** — Build a retrieval-augmented generation pipeline using open-source or API-based models on CHTC.
- **Resource etiquette** — Follow best practices for shared infrastructure, manage disk usage, and monitor your jobs.

#### Prerequisites

This workshop assumes you have a **fundamental ML/AI background**. Specifically, you should be comfortable with:

- **Python** — writing scripts, using packages like pandas and NumPy. New to Python? See the [Intro to Python](https://uw-madison-datascience.github.io/ML-X-Nexus/Learn/Workshops/Intro-Python_Gapminder.html) workshop.
- **Core ML/AI concepts** — train/test splits, overfitting, loss functions, hyperparameters. New to ML/AI? See the [Intro to Machine Learning](https://uw-madison-datascience.github.io/ML-X-Nexus/Learn/Workshops/Intro-ML_Sklearn.html) workshop.
- **Training a model** — you've trained at least one model in any framework (scikit-learn, PyTorch, TensorFlow, XGBoost, etc.).
- **Command line basics** — navigating directories, running commands in a terminal.

No prior CHTC or HTCondor experience is required — that's what this workshop teaches.

[workbench]: https://carpentries.github.io/sandpaper-docs
