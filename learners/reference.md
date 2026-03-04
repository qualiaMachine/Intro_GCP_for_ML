---
title: Reference
---

## Glossary

This glossary covers the GCP and ML terms used in this workshop. Refer back here whenever you hit an unfamiliar term during the episodes.

### Cloud Computing Basics

Virtual Machine (VM)
: A software-based computer that runs on Google's Compute Engine infrastructure. Each Vertex AI Workbench notebook is backed by a Compute Engine VM.

Instance
: A running VM in the cloud. In GCP, instances are defined by machine families (e.g., N2, C2, A2, A3) and can be customized for CPU, memory, and GPU needs.

Container
: A lightweight, isolated environment that packages code and dependencies together. Vertex AI training jobs run inside containers built from prebuilt Docker images.

Docker
: The most common container platform. Many GCP ML environments (like TensorFlow or PyTorch training images) are Docker containers hosted on Artifact Registry.

### GCP Services

Google Cloud Console
: The web-based interface for managing GCP resources, available at [console.cloud.google.com](https://console.cloud.google.com/). This is where you create buckets, launch notebooks, monitor training jobs, check billing, and manage permissions.

Compute Engine (GCE)
: The core infrastructure service that provides customizable VMs. Workbench notebooks and training jobs run on Compute Engine under the hood.

Vertex AI
: Google's unified ML platform — training jobs, tuning, notebooks, deployment, and more. The main service used throughout this workshop.

Cloud Storage (GCS)
: GCP's object storage service for datasets, models, and artifacts. The direct counterpart to AWS S3.

Bucket
: A top-level container in Cloud Storage that holds files (objects). Accessed via URIs like `gs://your-bucket-name/path/to/file.csv`.

GCS URI
: The unique path referencing an object in a Cloud Storage bucket. Example: `gs://ml-project-dataset/train.csv`.

Persistent Disk (PD)
: Block storage attached to VMs, including Workbench notebooks. Retains data between VM reboots — used for local datasets, checkpoints, or outputs.

Cloud Shell
: A browser-based terminal built into the Google Cloud Console (click the **>\_** icon in the top-right toolbar). It comes with `gcloud` pre-installed and already authenticated to your project.

### Access and Billing

IAM (Identity and Access Management)
: GCP's permission system. Defines who (user, service account, or group) can access which resources and at what privilege level.

Service Account
: A special Google identity used by applications to access GCP resources. Your Workbench notebook uses a service account to read from Cloud Storage and launch training jobs.

Quotas and Limits
: Default usage caps (e.g., max GPUs per region). Quotas can be increased through the console — understanding them helps prevent job failures.

Billing Alerts
: GCP's Budgets & Alerts feature tracks project spending and sends notifications when costs exceed thresholds.

### Vertex AI Workbench and ML/AI Workflows

Vertex AI Workbench
: A managed Jupyter notebook environment on Compute Engine. Used to run experiments and coordinate ML/AI workflows interactively.

Controller
: In this workshop, the notebook acts as the controller — it configures and submits training, tuning, and evaluation jobs via the Vertex AI SDK rather than running heavy computation locally.

Vertex AI Custom Job
: A managed training job that runs your code on dedicated Compute Engine instances. Equivalent to an AWS SageMaker Training Job.

Hyperparameter Tuning Job
: A Vertex AI service that searches for the best model configuration by running multiple trials with different hyperparameter sets.

Model Registry
: Stores trained models for versioning, deployment, and comparison across experiments.

Endpoint
: A deployed model that serves predictions through Vertex AI Prediction.

### Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG)
: A pattern where an LLM answers questions by first retrieving relevant passages from a corpus, then generating a response grounded in those passages. This reduces hallucination and allows citation of sources.

Chunking
: The process of breaking a large document into smaller, overlapping text segments so that each segment can be independently embedded and retrieved. Common strategies include fixed-character, sentence-level, and paragraph-level chunking.

Embedding
: A dense numerical vector (array of floats) that represents the semantic meaning of a piece of text. Texts with similar meanings produce vectors that are close together in the embedding space, enabling search by meaning rather than exact keywords.

Vector Similarity / Cosine Similarity
: A measure of how similar two embedding vectors are. Cosine similarity ranges from -1 (opposite) to 1 (identical direction). In RAG, it's used to rank which corpus chunks are most relevant to a user's query.

Nearest Neighbors
: An algorithm that finds the data points (embeddings) closest to a given query point in vector space. Used in RAG to retrieve the top-k most relevant chunks for a user's question.

Grounding
: The practice of constraining an LLM's response to information present in the retrieved context, rather than allowing it to generate answers from its general training data. Grounding reduces hallucination and improves factual accuracy.

Task Type (Embedding)
: A parameter passed to embedding models like `gemini-embedding-001` that tells the model to optimize its output for a specific use case. Common values: `RETRIEVAL_DOCUMENT` (for corpus text being indexed) and `RETRIEVAL_QUERY` (for user questions being searched).

## Additional Resources

- [Instances for ML on GCP](instances-for-ML.html) — guide to choosing machine types and GPUs
- [UW-Madison Cloud Resources](uw-madison-cloud-resources.html) — discounts, credits, and campus support for UW researchers
- [Using a GitHub PAT in Vertex AI](github-pat.html) — pushing/pulling code from Workbench notebooks
