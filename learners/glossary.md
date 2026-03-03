---
title: Glossary
---

This glossary covers the GCP and ML terms used in this workshop. You don't need to memorize these upfront — refer back here whenever you hit an unfamiliar term during the episodes.


### Cloud Computing Basics

* **Virtual Machine (VM)**: A software-based computer that runs on Google's Compute Engine infrastructure. Each Vertex AI Workbench notebook is backed by a Compute Engine VM.
* **Instance**: A running VM in the cloud. In GCP, instances are defined by machine families (e.g., **N2**, **C2**, **A2**, **A3**) and can be customized for CPU, memory, and GPU needs.
* **Container**: A lightweight, isolated environment that packages code and dependencies together. Vertex AI training jobs run inside containers built from prebuilt Docker images.
* **Docker**: The most common container platform. Many GCP ML environments (like TensorFlow or PyTorch training images) are Docker containers hosted on **Artifact Registry**.


### GCP Services

* **Compute Engine (GCE)**: The core infrastructure service that provides customizable VMs. Workbench notebooks and training jobs run on Compute Engine under the hood.
* **Vertex AI**: Google's unified ML platform — training jobs, tuning, notebooks, deployment, and more. The main service used throughout this workshop.
* **Cloud Storage (GCS)**: GCP's object storage service for datasets, models, and artifacts. The direct counterpart to AWS S3.
* **Bucket**: A top-level container in Cloud Storage that holds files (objects). Accessed via URIs like `gs://your-bucket-name/path/to/file.csv`.
* **GCS URI**: The unique path referencing an object in a Cloud Storage bucket. Example: `gs://ml-project-dataset/train.csv`.
* **Persistent Disk (PD)**: Block storage attached to VMs, including Workbench notebooks. Retains data between VM reboots — used for local datasets, checkpoints, or outputs.


### Access and Billing

* **IAM (Identity and Access Management)**: GCP's permission system. Defines who (user, service account, or group) can access which resources and at what privilege level.
* **Service Account**: A special Google identity used by applications to access GCP resources. Your Workbench notebook uses a service account to read from Cloud Storage and launch training jobs.
* **Quotas and Limits**: Default usage caps (e.g., max GPUs per region). Quotas can be increased through the console — understanding them helps prevent job failures.
* **Billing Alerts**: GCP's **Budgets & Alerts** feature tracks project spending and sends notifications when costs exceed thresholds.


### Vertex AI Workbench and ML Workflows

* **Vertex AI Workbench**: A managed Jupyter notebook environment on Compute Engine. Used to run experiments and coordinate ML workflows interactively.
* **Controller**: In this workshop, the notebook acts as the controller — it configures and submits training, tuning, and evaluation jobs via the Vertex AI SDK rather than running heavy computation locally.
* **Vertex AI Custom Job**: A managed training job that runs your code on dedicated Compute Engine instances. Equivalent to an AWS SageMaker Training Job.
* **Hyperparameter Tuning Job**: A Vertex AI service that searches for the best model configuration by running multiple trials with different hyperparameter sets.
* **Model Registry**: Stores trained models for versioning, deployment, and comparison across experiments.
* **Endpoint**: A deployed model that serves predictions through Vertex AI Prediction.
