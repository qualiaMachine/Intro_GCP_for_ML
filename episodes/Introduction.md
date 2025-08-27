---
title: "Overview of Google Cloud Vertex AI"
teaching: 10
exercises: 1
---

Google Cloud Vertex AI is a unified machine learning (ML) platform that enables users to build, train, tune, and deploy models at scale without needing to manage underlying infrastructure. By integrating data storage, training, tuning, and deployment workflows into one managed environment, Vertex AI supports researchers and practitioners in focusing on their ML models while leveraging Google Cloud’s compute and storage resources.

::::::::::::::::::::::::::::::::::::: questions

- What problem does Google Cloud Vertex AI aim to solve?  
- How does Vertex AI simplify machine learning workflows compared to running them on your own?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand the basic purpose of Vertex AI in the ML lifecycle.  
- Recognize how Vertex AI reduces infrastructure and orchestration overhead.  

::::::::::::::::::::::::::::::::::::::::::::::::

### Why use Vertex AI for machine learning?

Vertex AI provides several advantages that make it an attractive option for research and applied ML:

- **Streamlined ML/AI Pipelines**: Traditional HPC/HTC environments often require researchers to split workflows into many batch jobs, manually handling dependencies and orchestration. Vertex AI reduces this overhead by managing the end-to-end pipeline (data prep, training, evaluation, tuning, and deployment) within a single environment, making it easier to iterate and scale ML experiments.

- **Flexible compute options**: Vertex AI lets you select the right hardware for your workload:
  - **CPU (e.g., n1-standard-4, e2-standard-8)**: Good for small datasets, feature engineering, and inference tasks.  
  - **GPU (e.g., NVIDIA T4, V100, A100)**: Optimized for deep learning training and large-scale experimentation.  
  - **Memory-optimized machine types (e.g., m1-ultramem)**: Useful for workloads requiring large in-memory datasets, such as transformer models.  

- **Parallelized training and tuning**: Vertex AI supports distributed training across multiple nodes and automated hyperparameter tuning (Bayesian or grid search). This makes it easier to explore many configurations with minimal custom code while leveraging scalable infrastructure.

- **Custom training support**: Vertex AI includes built-in algorithms and frameworks (e.g., scikit-learn, XGBoost, TensorFlow, PyTorch), but it also supports custom containers. Researchers can bring their own scripts or Docker images to run specialized workflows with full control.

- **Cost management and monitoring**: Google Cloud provides detailed cost tracking and monitoring via the Billing console and Vertex AI dashboard. Vertex AI also integrates with Cloud Monitoring to help track resource usage. With careful configuration, training 100 small-to-medium models (logistic regression, random forests, or lightweight neural networks on datasets under 10GB) can cost under $20, similar to AWS.

In summary, Vertex AI is Google Cloud’s managed machine learning platform that simplifies the end-to-end ML lifecycle. It eliminates the need for manual orchestration in research computing environments by offering integrated workflows, scalable compute, and built-in monitoring. With flexible options for CPUs, GPUs, and memory-optimized hardware, plus strong support for both built-in and custom training, Vertex AI enables researchers to move quickly from experimentation to production while keeping costs predictable and manageable.


::::::::::::::::::::::::::::::::::::: challenge

### Infrastructure Choices for ML  
At your institution (or in your own work), what infrastructure options are currently available for running ML experiments?  
- Do you typically use a laptop/desktop, HPC cluster, or cloud?  
- What are the advantages and drawbacks of your current setup compared to a managed service like Vertex AI?  
- If you could offload one infrastructure challenge (e.g., provisioning GPUs, handling dependencies, monitoring costs), what would it be and why?  

Take 3–5 minutes to discuss with a partner or share in the workshop chat.  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Vertex AI simplifies ML workflows by integrating data, training, tuning, and deployment in one managed platform.  
- It reduces the need for manual orchestration compared to traditional research computing environments.  
- Cost monitoring and resource tracking help keep cloud usage affordable for research projects.  

::::::::::::::::::::::::::::::::::::::::::::::::
