---
title: "Notebooks as Controllers"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do you set up and use Vertex AI Workbench notebooks for machine learning tasks?  
- How can you manage compute resources efficiently using a “controller” notebook approach in GCP?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Describe how to use Vertex AI Workbench notebooks for ML workflows.  
- Set up a Jupyter-based Workbench instance as a controller to manage compute tasks.  
- Use the Vertex AI SDK to launch training and tuning jobs on scalable instances.  

::::::::::::::::::::::::::::::::::::::::::::::::

## Setting up our notebook environment
Google Cloud Vertex AI provides a managed environment for building, training, and deploying machine learning models. In this episode, we’ll set up a **Vertex AI Workbench notebook instance**—a Jupyter-based environment hosted on GCP that integrates seamlessly with other Vertex AI services.  

### Using the notebook as a controller
The notebook instance functions as a *controller* to manage more resource-intensive tasks. By selecting a modest machine type (e.g., `n1-standard-4`), you can perform lightweight operations locally in the notebook while using the **Vertex AI Python SDK** to launch compute-heavy jobs on larger machines (e.g., GPU-accelerated) when needed.  

This approach minimizes costs while giving you access to scalable infrastructure for demanding tasks like model training, batch prediction, and hyperparameter tuning.  

We’ll follow these steps to create our first Vertex AI Workbench notebook:

#### 1. Navigate to Vertex AI Workbench
- In the Google Cloud Console, search for **Vertex AI Workbench**.  
- Pin it to your navigation bar for quick access.  

#### 2. Create a new notebook instance
- Click **New Notebook**.  
- Choose **Managed Notebooks** (recommended for workshops and shared environments).  
- **Notebook name**: Use a naming convention like `yourname-explore-vertexai`.  
- **Machine type**: Select a small machine (e.g., `n1-standard-4`) to act as the controller.  
  - This keeps costs low while you delegate heavy lifting to Vertex AI training jobs.  
  - For guidance on common machine types for ML procedures, refer to our supplemental [Instances for ML on GCP](../instances-for-ML.html).  
- **GPUs**: Leave disabled for now (training jobs will request them separately).  
- **Permissions**: The project’s default service account is usually sufficient. It must have access to GCS and Vertex AI.  
- **Networking and encryption**: Leave default unless required by your institution.  
- **Labels**: Add labels for cost tracking (e.g., `purpose=workshop`, `owner=yourname`).  

Once created, your notebook instance will start in a few minutes. When its status is **Running**, you can open JupyterLab and begin working.  

### Managing training and tuning with the controller notebook
In the following episodes, we’ll use the **Vertex AI Python SDK (`google-cloud-aiplatform`)** from this notebook to submit compute-heavy tasks on more powerful machines. Examples include:  

- **Training a model**: Submit a training job to Vertex AI with a higher-powered instance (e.g., `n1-highmem-32` or GPU-backed machines).  
- **Hyperparameter tuning**: Configure and submit a tuning job, allowing Vertex AI to manage multiple parallel trials automatically.  

This pattern keeps costs low by running your notebook on a modest VM while only incurring charges for larger resources when they’re actively in use.  

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge: Notebook Roles

Your university provides different compute options: laptops, on-prem HPC, and GCP.  

- What role does a **Vertex AI Workbench notebook** play compared to an HPC login node or a laptop-based JupyterLab?  
- Which tasks should stay in the notebook (lightweight control, visualization) versus being launched to larger cloud resources?  

:::::::::::::::: solution

The notebook serves as a lightweight control plane.  
- Like an HPC login node, it’s not meant for heavy computation.  
- Suitable for small preprocessing, visualization, and orchestrating jobs.  
- Resource-intensive tasks (training, tuning, batch jobs) should be submitted to scalable cloud resources (GPU/large VM instances) via the Vertex AI SDK.  

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints 

- Use a small Vertex AI Workbench notebook instance as a controller to manage larger, resource-intensive tasks.  
- Submit training and tuning jobs to scalable instances using the Vertex AI SDK.  
- Labels help track costs effectively, especially in shared or multi-project environments.  
- Vertex AI Workbench integrates directly with GCS and Vertex AI services, making it a hub for ML workflows.  

::::::::::::::::::::::::::::::::::::::::::::::::
