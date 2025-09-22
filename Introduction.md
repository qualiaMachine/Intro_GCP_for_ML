---
title: "Overview of Google Cloud for Machine Learning"
teaching: 10
exercises: 1
---

::::::::::::::::::::::::::::::::::::: questions

- What problem does GCP aim to solve for ML researchers?  
- How does using a notebook as a controller help organize ML workflows in the cloud?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand the basic role of GCP in supporting ML research.  
- Recognize how a notebook can serve as a controller for cloud resources.  

::::::::::::::::::::::::::::::::::::::::::::::::

Google Cloud Platform (GCP) provides the basic building blocks researchers need to run machine learning (ML) experiments at scale. Instead of working only on your laptop or a high-performance computing (HPC) cluster, you can spin up compute resources on demand, store datasets in the cloud, and run notebooks that act as a "controller" for larger training and tuning jobs.  

This workshop focuses on *using a simple notebook environment as the control center* for your ML workflow. We will not rely on Google’s fully managed Vertex AI platform, but instead show how to use core GCP services (Compute Engine, storage buckets, and SDKs) so you can build and run experiments from scratch.

### Why use GCP for machine learning?

GCP provides several advantages that make it a strong option for applied ML:

- **Flexible compute**: You can choose the hardware that fits your workload:  
  - **CPUs** for lightweight models, preprocessing, or feature engineering.  
  - **GPUs** (e.g., NVIDIA T4, V100, A100) for training deep learning models.  
  - **High-memory machines** for workloads that need large datasets in memory.  

- **Data storage and access**: Google Cloud Storage (GCS) buckets act like S3 on AWS — an easy way to store and share datasets between experiments and collaborators.  

- **From scratch workflows**: Instead of depending on a fully managed ML service, you bring your own frameworks (PyTorch, TensorFlow, scikit-learn, etc.) and run your code the same way you would on your laptop or HPC cluster, but with scalable cloud resources.  

- **Cost visibility**: Billing dashboards and project-level budgets make it easier to track costs and stay within research budgets.  

In short, GCP provides infrastructure that you control from a notebook environment, allowing you to build and run ML workflows just as you would locally, but with access to scalable hardware and storage.

::::::::::::::::::::::::::::::::::::: challenge

### Comparing infrastructures  
Think about your current research setup:  
- Do you mostly use your laptop, HPC cluster, or cloud for ML experiments?  
- What benefits would running a cloud-based notebook controller give you?  
- If you could offload one infrastructure challenge (e.g., installing GPU drivers, managing storage, or setting up environments), what would it be and why?  

Take 3–5 minutes to discuss with a partner or share in the workshop chat.  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- GCP provides the core building blocks (compute, storage, networking) for ML research.  
- A notebook can act as a controller to organize cloud workflows and keep experiments reproducible.  
- Using raw infrastructure instead of a fully managed platform gives researchers flexibility while still benefiting from scalable cloud resources.  

::::::::::::::::::::::::::::::::::::::::::::::::
