---
title: "Overview of Google Cloud for Machine Learning"
teaching: 10
exercises: 2
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

This workshop focuses on *using a simple notebook environment as the control center* for your ML workflow. You write and debug code in a Vertex AI Workbench notebook, then submit training and tuning jobs to Vertex AI's compute infrastructure — keeping your notebook lightweight while GCP handles the heavy lifting.

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

### A note on cloud costs

Cloud computing is not free — you pay for the compute, storage, and API calls you use. But it's worth putting those costs in context:

- **Hardware is expensive and evolves fast.** A single A100 GPU costs ~$15,000 and is outdated within a few years. Cloud lets you rent the latest hardware by the hour without a capital purchase or a depreciation schedule.
- **Managing hardware takes people.** On-premises clusters require staff for procurement, installation, driver updates, cooling, and security. Cloud offloads that operational burden so researchers can focus on research.
- **You pay only for what you use.** Stop a VM and the meter stops. This is especially valuable for bursty research workloads — a week of intensive training followed by months of writing.
- **Budgets and alerts keep you safe.** GCP billing dashboards, budget alerts, and quotas help prevent surprise bills. We cover cleanup practices in [Episode 9](09-Resource-management-cleanup.md).

The key is to be intentional: choose the right machine size, stop resources when idle, and monitor spending. We'll reinforce these habits throughout the workshop.

::::::::::::::::::::::::::::::::::::: callout

### For UW-Madison researchers

If you're at UW-Madison, the university offers several resources that can significantly reduce your cloud costs:

- **Reduced overhead on grants** — The [Cloud Computing Pilot](https://rsp.wisc.edu/proposalprep/cloudComputeInfo.cfm) drops F&A overhead from 55.5% to **26%** on cloud expenses when using a UW-provisioned account, saving ~$2,950 per $10,000 spent.
- **NIH STRIDES discounts** — Additional pricing reductions for NIH-funded researchers, layered on top of UW rates.
- **Google Cloud Research Credits** — Up to **$5,000** in free credits for faculty and postdocs ($1,000 for PhD students). [Apply here](https://edu.google.com/intl/ALL_us/programs/credits/research/).
- **Free on-campus GPUs** — [CHTC](https://chtc.cs.wisc.edu/) provides access to hundreds of GPUs (including A100s) at no cost. Great for workloads that don't need cloud-specific services.
- **Support** — The [Public Cloud Team](mailto:cloud-services@cio.wisc.edu) offers weekly office hours and architecture consultations. The [ML+X community](https://hub.datascience.wisc.edu/communities/mlx/) holds monthly meetings on ML/AI topics.

See the [UW-Madison Cloud Resources](../uw-madison-cloud-resources.html) page for the full list of discounts, contacts, and how to request a UW cloud account.

::::::::::::::::::::::::::::::::::::::::::::::::

### GCP and Vertex AI: a quick orientation

Google Cloud has many products and brand names. The table below maps the key terms you'll encounter in this workshop to what they actually do.

| Term | What it is | Where you'll see it |
|------|-----------|-------------------|
| **GCP (Google Cloud Platform)** | The overall cloud platform — compute, storage, networking, and managed services. | Everything in this workshop runs on GCP. |
| **Vertex AI** | Google's umbrella brand for ML/AI services. It includes Workbench notebooks, training jobs, hyperparameter tuning, Model Garden, and more. | Episodes 3–10 all use Vertex AI services. |
| **Vertex AI Workbench** | A managed Jupyter notebook environment that runs on a Compute Engine VM. Think of it as "Jupyter in the cloud" with GCP authentication built in. | Episode 3 — where we create our notebook. |
| **Cloud Storage (GCS)** | Object storage for datasets, model artifacts, and scripts. Similar to AWS S3. | Episode 2 — where we upload training data. |
| **Compute Engine** | Virtual machines (VMs) you can configure with CPUs, GPUs, or TPUs. Workbench notebooks run on Compute Engine VMs under the hood. | Episodes 6–8 — the machines that run training jobs. |
| **Gemini** | Google's family of large language models (LLMs), used for text generation, embeddings, and multimodal tasks. Accessed through the Vertex AI API. | Episode 10 — RAG pipeline uses Gemini for generation and embeddings. |
| **Model Garden** | A catalog of foundation models (Google and third-party) available through Vertex AI. | Episode 10 — where you can browse embedding and generation models. |

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
