---
title: "Overview of Google Cloud for Machine Learning"
teaching: 10
exercises: 1
---

::::::::::::::::::::::::::::::::::::: questions

- What problem does GCP aim to solve for ML researchers?  
- How does using a notebook as a controller help organize ML workflows in the cloud?  
- How does GCP compare to AWS for ML workflows?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand the basic role of GCP in supporting ML research.  
- Recognize how a notebook can serve as a controller for cloud resources.  
- Compare GCP and AWS approaches to building and managing ML workflows.  

::::::::::::::::::::::::::::::::::::::::::::::::

Google Cloud Platform (GCP) provides the basic building blocks researchers need to run machine learning (ML) experiments at scale. Instead of working only on your laptop or a high-performance computing (HPC) cluster, you can spin up compute resources on demand, store datasets in the cloud, and run low-cost notebooks that act as a "controller" for larger training and tuning jobs.  

This workshop focuses on *using a simple notebook environment as the control center* for your ML workflow. We will not rely on Google's fully managed Vertex AI platform, but instead show how to use core GCP services (Compute Engine, storage buckets, and SDKs) so you can build and run experiments from scratch.  

### Why use GCP for machine learning?

GCP provides several advantages that make it a strong option for applied ML:

- **Flexible compute**: You can choose the hardware that fits your workload:  
  - **CPUs** for lightweight models, preprocessing, or feature engineering.  
  - **GPUs** (e.g., NVIDIA T4, V100, A100) for training deep learning models.  
  - **TPUs (Tensor Processing Units)** for TensorFlow or JAX-based deep learning. TPUs are custom Google hardware optimized for matrix operations and can provide strong performance and energy efficiency for compatible workloads. Google has reported better performance-per-watt compared to GPUs in many TensorFlow benchmarks, though *these gains depend heavily on model type and implementation*.  
    Historically, TPU support has been limited for PyTorch users, and while Google is improving PyTorch integration, the TPU ecosystem still works best for TensorFlow and JAX workflows.

- **Data storage and access**: Google Cloud Storage (GCS) buckets act like S3 on AWS — an easy way to store and share datasets between experiments and collaborators.  

- **From scratch workflows**: Instead of depending on a fully managed ML service, you bring your own frameworks (PyTorch, TensorFlow, scikit-learn, etc.) and run your code the same way you would on your laptop or HPC cluster, but with scalable cloud resources.  

- **Cost visibility**: Billing dashboards and project-level budgets make it easier to track costs and stay within research budgets.  

- **Sustainability focus**: Google aims to operate entirely on *carbon-free energy by 2030*. Combined with the TPU's focus on efficient matrix computation, this gives GCP a potential edge for researchers interested in energy-conscious ML — though *real-world energy efficiency varies by workload and utilization*.  

In short, GCP provides infrastructure that you control from a notebook environment, allowing you to build and run ML workflows just as you would locally, but with access to scalable hardware and storage.


::::::::::::::::::::::::::::::::::::: callout

### What about AWS?

In many respects, GCP and AWS offer comparable capabilities for ML research. Both provide scalable compute, storage, and tooling to support everything from quick experiments to production pipelines.  
AWS typically offers a broader range of GPU and CPU instance types, along with mature managed services like SageMaker and tighter integration with enterprise infrastructure. GCP, on the other hand, emphasizes the use of TensorFlow and JAX, and the availability of TPUs — which may offer energy advantages for certain workloads.  

Ultimately, the choice often comes down to framework preference, familiarity, and existing resources, rather than major functional differences between the two platforms.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Comparing infrastructures  
Think about your current research setup:  
- Do you mostly use your laptop, HPC cluster, AWS, or GCP for ML experiments?  
- Which environment feels most transparent for understanding costs and reproducibility?  
- If you could offload one infrastructure challenge (e.g., installing GPU drivers, managing storage, or setting up environments), what would it be and why?  

Take 3–5 minutes to discuss with a partner or share in the workshop chat.  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- GCP and AWS both provide the essential components for running ML workloads at scale.  
- GCP emphasizes simplicity, open frameworks, and TPU access; AWS offers broader hardware and automation options.  
- TPUs are efficient for TensorFlow and JAX, but GPU-based workflows (common on AWS) remain more flexible across frameworks.  
- Both platforms now provide strong cost tracking and sustainability tools, with only minor differences in interface and ecosystem integration.  
- Using a notebook as a controller provides reproducibility and helps manage compute and storage resources consistently across clouds.  

::::::::::::::::::::::::::::::::::::::::::::::::
