---
title: Glossary (GCP)
---

Understanding the terminology used in cloud computing and GCP is half the battle when working with Vertex AI and Workbench. Familiarity with these key concepts will help you navigate Google Cloud services, configure machine learning workflows, and troubleshoot issues more efficiently.  

We encourage you to briefly study this glossary before the workshop and refer back to it as needed. While we'll go over these terms throughout the workshop, early exposure will make the hands-on exercises smoother and faster.  


### Cloud Compute Essentials  

* **Serverless**: A way of running code without managing infrastructure. The cloud provider handles provisioning, scaling, and maintenance automatically, and you only pay when your code runs. In GCP, examples include **Cloud Functions**, **Cloud Run**, and **Vertex AI Predictions**, which can scale to zero when idle.  
* **Virtual Machine (VM)**: A software-based computer that runs on Google’s Compute Engine infrastructure. Each **Vertex AI Workbench notebook** is ultimately backed by a Compute Engine VM, even if the environment looks fully managed.  
* **Instance**: A running VM in the cloud. In GCP, instances are defined by machine families (e.g., **N2**, **C2**, **A2**, **A3**) and can be customized for CPU, memory, and GPU needs. These are the same instance types you select for Vertex AI Workbench notebooks.  
* **Container**: A lightweight, isolated environment that packages code and dependencies together. Containers ensure consistent execution and are the foundation of services like **Vertex AI Workbench** (notebook containers), **Cloud Run**, and **Kubernetes Engine**.  
* **Docker**: The most common container platform used for building, shipping, and running containerized applications. Many GCP ML environments (like TensorFlow or PyTorch Workbench images) are built as Docker containers hosted on **Google Container Registry (GCR)** or **Artifact Registry**.  
* **Elasticity**: The ability of cloud resources to scale up or down automatically based on workload. GCP provides elasticity through **autoscaling managed instance groups**, **Kubernetes Engine**, and **Vertex AI training services**.  

### GCP General  

* **Compute Engine (GCE)**: The core infrastructure service that provides customizable VMs. Vertex AI Workbench notebooks, training jobs, and many ML services are built on top of Compute Engine.  
* **Vertex AI**: A unified machine learning platform that integrates model training, tuning, deployment, and monitoring. It supports managed notebooks, training jobs, pipelines, and AutoML but can also run fully custom ML code.  
* **Auto Scaling**: A feature that automatically adjusts the number of VM instances in a managed instance group based on utilization metrics such as CPU or memory.  


### Account Governance and Security  

* **IAM (Identity and Access Management)**: GCP’s permission management system. It defines who (user, service account, or group) can access what resources (Vertex AI, Storage, Compute Engine) and with what level of privilege.  
* **Service Account**: A special Google identity used by applications and services to access GCP resources securely. For example, a Vertex AI Workbench notebook uses a service account to read data from Cloud Storage or launch training jobs.  
  - **Relation to Cloud Storage Policies**: Service accounts grant programmatic access at the project or resource level, while **bucket-level permissions** control access to individual Cloud Storage buckets. Both must align for data access to work.  
* **Bucket Policy (Cloud Storage IAM Policy)**: Defines who can read, write, or manage objects in a Google Cloud Storage bucket. These policies are project- and bucket-scoped and often reference service accounts.  
* **Access Control Lists (ACLs)**: A legacy way to manage fine-grained access for specific Cloud Storage objects. ACLs are now largely superseded by IAM policies but can still appear in legacy datasets.  
* **Organization Policy Service**: A GCP feature for defining constraints and policies across multiple projects (e.g., restricting region usage or service types). Similar to **AWS Organizations**, it supports centralized governance and billing.  
* **Quotas and Limits**: GCP places default usage caps (e.g., maximum number of CPUs or GPUs per region). Quotas can be increased through the **Quota Management Console**, and understanding them helps prevent resource allocation failures.  
* **Billing Alerts**: GCP provides **Budgets & Alerts** to track project spending and receive email or Pub/Sub notifications when costs exceed thresholds.  


### Data Storage and Management  

* **Cloud Storage (GCS)**: GCP’s object storage service for datasets, models, and artifacts. It’s highly scalable and the direct counterpart to AWS S3.  
* **Bucket**: A top-level container within Cloud Storage that holds data files (objects). Each file can be accessed via a unique URI in the form `gs://your-bucket-name/path/to/file.csv`.  
* **GCS URI (Object URI)**: The unique path referencing an object in a Cloud Storage bucket, typically used by Vertex AI and Workbench for loading or saving data. Example: `gs://ml-project-dataset/train.csv`.  
* **Persistent Disk (PD)**: Block storage volumes attached to VMs, including Workbench notebooks. They retain data between VM reboots and can be used to store local datasets, checkpoints, or outputs.  
* **Filestore / Cloud Storage FUSE**: Options for mounting network file systems or Cloud Storage buckets directly to your notebook’s filesystem.  


### Vertex AI Workbench and Machine Learning Workflows  

* **Vertex AI Workbench**: A managed Jupyter notebook environment built on top of Compute Engine. It comes in **Managed** and **User-Managed** modes and is used to run ML experiments and manage data workflows interactively.  
* **Workbench Notebook Instance**: The actual VM running your notebook container. You configure its machine type (CPU/GPU), disk size, and region just like any other VM.  
* **Controller**: In this workshop, the notebook itself acts as the controller — it configures and runs training, tuning, and evaluation jobs through the Vertex AI SDK rather than performing all computation inside the notebook runtime.  
* **Vertex AI Custom Job**: A managed training job that executes your custom training code on dedicated Compute Engine instances. It’s equivalent to a SageMaker Training Job in AWS.  
* **Hyperparameter Tuning Job**: A Vertex AI service that automatically searches for the best model configuration by evaluating multiple trials with different hyperparameter sets.  
* **Model Registry**: Stores trained models for versioning, deployment, and comparison across experiments.  
* **Endpoint (for Deployment)**: A deployed instance of a trained model that serves predictions through Vertex AI Prediction.  