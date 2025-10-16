---
title: "Notebooks as Controllers"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do you set up and use Vertex AI Workbench notebooks for machine learning tasks?  
- How can you manage compute resources efficiently using a "controller" notebook approach in GCP?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Describe how to use Vertex AI Workbench notebooks for ML workflows.  
- Set up a Jupyter-based Workbench instance as a controller to manage compute tasks.  
- Use the Vertex AI SDK to launch training and tuning jobs on scalable instances.  

::::::::::::::::::::::::::::::::::::::::::::::::

## Setting up our notebook environment
Google Cloud Workbench provides JupyterLab-based environments that can be used to orchestrate machine learning workflows. In this workshop, we will use a **Workbench Instance**—the recommended option going forward, as other Workbench environments are being deprecated.  

> Workbench Instances come with JupyterLab 3 pre-installed and are configured with GPU-enabled ML frameworks (TensorFlow, PyTorch, etc.), making it easy to start experimenting without additional setup. Learn more in the [Workbench Instances documentation](https://cloud.google.com/vertex-ai/docs/workbench/instances/introduction?_gl=1*r0g0e9*_ga*MTczMzg4NDE1OC4xNzU4MzEyMTE0*_ga_WH2QY8WWF5*czE3NTg1NTczMzkkbzMkZzEkdDE3NTg1NjIxNzgkajI3JGwwJGgw).  

### Using the notebook as a controller
The notebook instance functions as a *controller* to manage more resource-intensive tasks. By selecting a modest machine type (e.g., `n1-standard-4`), you can perform lightweight operations locally in the notebook while using the **Vertex AI Python SDK** to launch compute-heavy jobs on larger machines (e.g., GPU-accelerated) when needed.  

This approach minimizes costs while giving you access to scalable infrastructure for demanding tasks like model training, batch prediction, and hyperparameter tuning.  

We will follow these steps to create our first Workbench Instance:

#### 1. Navigate to Workbench
- In the Google Cloud Console, search for "Workbench."  
- Click the "Instances" tab (this is the supported path going forward).  
- Pin Workbench to your navigation bar for quick access.  

#### 2. Create a new Workbench Instance

##### Initial settings
- Click **Create New** near the top of the Workbench page
- **Name**: For this workshop, we can use the following naming convention to easily locate our notebooks: `teamname-yourname-purpose` (e.g., sinkorswim-johndoe-train)
- **Region**: Choose the same region as your storage bucket (e.g., `us-central1`). This avoids cross-region transfer charges and keeps data access latency low.
    - If you are unsure, check your bucket's location in the Cloud Storage console (click the bucket name → look under "Location").
- **Zone:** `us-central1-a` (or another zone in `us-central1`, like `-b` or `-c`)  
  - If capacity or GPU availability is limited in one zone, switch to another zone in the same region.
- **NVIDIA T4 GPU:** Leave unchecked for now  
  - We will request GPUs for training jobs separately. Attaching here increases idle costs.
- **Apache Spark and BigQuery Kernels:** Leave unchecked  
  - Enable only if you specifically need Spark or BigQuery notebooks; otherwise, it adds unnecessary images.
- **Network in this project:** Required selection  
  - This option must be selected; shared environments do not allow using external or default networks.  
  - This ensures your instance connects to the shared VPC for the workshop.
- **Network / Subnetwork:**  Leave as pre-filled.
![Notebook settings (part 1)](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/new-instance-settings1.jpg){alt="Notebook settings (part1)"}

##### Advanced settings: Details (tagging)

- **IMPORTANT:** Open the "Advanced optoins menu next
  -  **Labels (required for cost tracking):**  Under the Details menu, add the following tags so that you can track the total cost of your activity on GCP later:
      - `Project = Team Name`
      - `Name = Your Name`
      - `Purpose = Notebook Purpose (train, tune, RAG, etc.)`
        
![Required tags for notebook.](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/new-instance-tags.jpg){alt="Screenshot showing required tags for notebook"}


##### Advanced settings: Machine Type 

- **Machine type**: Select a small machine (e.g., `e2-standard-2`) to act as the controller.  
  - This keeps costs low while you delegate heavy lifting to training jobs.  
  - For guidance on common machine types for ML, refer to [Instances for ML on GCP](../instances-for-ML.html).

- **Set idle shutdown**: To save on costs when you aren't doing anything in your notebook, lower the default idle shutdown time to **60 (minutes)**.

![Enable Idle Shutdown](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/new-instance-idleshutdown.jpg){alt="Set Idle Shutdown"}


##### Create notebook 
- Click **Create** to create the instance. Your notebook instance will start in a few minutes. When its status is "Running," you can open JupyterLab and begin working.  

### Managing training and tuning with the controller notebook
In the following episodes, we will use the **Vertex AI Python SDK (`google-cloud-aiplatform`)** from this notebook to submit compute-heavy tasks on more powerful machines. Examples include:  

- Training a model on a GPU-backed instance.  
- Running hyperparameter tuning jobs managed by Vertex AI.  

This pattern keeps costs low by running your notebook on a modest VM while only incurring charges for larger resources when they are actively in use.  

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge: Notebook Roles

Your university provides different compute options: laptops, on-prem HPC, and GCP.  

- What role does a **Workbench Instance notebook** play compared to an HPC login node or a laptop-based JupyterLab?  
- Which tasks should stay in the notebook (lightweight control, visualization) versus being launched to larger cloud resources?  

:::::::::::::::: solution

The notebook serves as a lightweight control plane.  
- Like an HPC login node, it is not meant for heavy computation.  
- Suitable for small preprocessing, visualization, and orchestrating jobs.  
- Resource-intensive tasks (training, tuning, batch jobs) should be submitted to scalable cloud resources (GPU/large VM instances) via the Vertex AI SDK.  

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints 

- Use a small Workbench Instance notebook as a controller to manage larger, resource-intensive tasks.  
- Always navigate to the "Instances" tab in Workbench, since older notebook types are deprecated.  
- Choose the same region for your Workbench Instance and storage bucket to avoid extra transfer costs.  
- Submit training and tuning jobs to scalable instances using the Vertex AI SDK.  
- Labels help track costs effectively, especially in shared or multi-project environments.  
- Workbench Instances come with JupyterLab 3 and GPU frameworks preinstalled, making them an easy entry point for ML workflows.  
- Enable idle auto-stop to avoid unexpected charges when notebooks are left running.  

::::::::::::::::::::::::::::::::::::::::::::::::
