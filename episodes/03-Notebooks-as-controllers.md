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

## Using the notebook as a controller
The notebook instance functions as a *controller* to manage more resource-intensive tasks. By selecting a modest machine type (e.g., `n1-standard-4`), you can perform lightweight operations locally in the notebook while using the **Vertex AI Python SDK** to launch compute-heavy jobs on larger machines (e.g., GPU-accelerated) when needed.  

This approach minimizes costs while giving you access to scalable infrastructure for demanding tasks like model training, batch prediction, and hyperparameter tuning.  

We will follow these steps to create our first Workbench Instance:

### 1. Navigate to Workbench
- In the Google Cloud Console, search for "Workbench."  
- Click the "Instances" tab (this is the supported path going forward).  
- Pin Workbench to your navigation bar for quick access.  

### 2. Create a new Workbench Instance

#### Initial settings
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

#### Advanced settings: Details (tagging)

- **IMPORTANT:** Open the "Advanced optoins menu next
  -  **Labels (required for cost tracking):**  Under the Details menu, add the following tags (all lowercase) so that you can track the total cost of your activity on GCP later:
      - `project = teamname` (your team's name)
      - `name = name` (your name)
      - `purpose = train` (i.e., the notebook's overall purpose — train, tune, RAG, etc.)
        
![Required tags for notebook.](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/new-instance-tags.jpg){alt="Screenshot showing required tags for notebook"}

#### Advanced Settings: Environment 

While we won't modify environment settings during this workshop, it's useful to understand what these options control when creating or editing a Workbench Instance in Vertex AI Workbench.

All Workbench environments use JupyterLab 3 by default, with the latest NVIDIA GPU drivers, CUDA libraries, and Intel optimizations preinstalled.  
You can optionally select JupyterLab 4 (currently in preview) or provide a custom container image to run your own environment (for example, a Docker image containing specialized ML frameworks or dependencies).  
If needed, you can also specify a post-startup script stored in Cloud Storage (`gs://path/to/script.sh`) to automatically configure the instance (install packages, mount buckets, etc.) when it boots.  

See: [Vertex AI Workbench release notes](https://cloud.google.com/vertex-ai/docs/release-notes) for supported versions and base images.

> In short: your Workbench environment is a containerized JupyterLab session running on a Compute Engine VM. These options control the version, performance, storage, networking, and permissions of that underlying VM, even though the interface abstracts most of the complexity away.

#### Advanced settings: Machine Type 

- **Machine type**: Select a small machine (e.g., `e2-standard-2`) to act as the controller.  
  - This keeps costs low while you delegate heavy lifting to training jobs.  
  - For guidance on common machine types for ML, refer to [Instances for ML on GCP](../instances-for-ML.html).

- **Set idle shutdown**: To save on costs when you aren't doing anything in your notebook, lower the default idle shutdown time to **60 (minutes)**.

![Enable Idle Shutdown](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/new-instance-idleshutdown.jpg){alt="Set Idle Shutdown"}

#### Advanced Settings: Disks

Each Vertex AI Workbench instance uses **Persistent Disks (PDs)** to store your system files and data. You'll configure two disks when creating a notebook: a **boot disk** and a **data disk**. We'll leave these at their default settings, but it's useful to understand the settings for future work.

##### Boot Disk
Keep this fixed at **100 GB (Balanced PD)** — the default minimum.  
It holds the OS, JupyterLab, and ML libraries but not your datasets.  
Estimated cost: about **$10 / month (~$0.014 / hr)**.  
You rarely need to resize this, though you can increase to **150–200 GB** if you plan to install large environments, custom CUDA builds, or multiple frameworks.

##### Data Disk
This is where your datasets, checkpoints, and outputs live.  
Use a **Balanced PD** by default, or an **SSD PD** only for high-I/O workloads.  
A good rule of thumb is to allocate **≈ 2× your dataset size**, with a **minimum of 150 GB** and a **maximum of 1 TB**.  
For example:
- 20 GB dataset → 150 GB data disk (minimum)  
- 100 GB dataset → 200 GB data disk  
- Larger datasets → keep the full dataset in **Cloud Storage (`gs://`)** and copy only subsets locally.

> Persistent Disks can be resized anytime without downtime, so it’s best to start small and expand when needed.

##### Deletion behavior
The 'Delete to trash' option is **unchecked by default**, which is what you want.  
When left unchecked, deleted files are removed immediately, freeing up disk space right away.  
If you check this box, files will move to the system trash instead — meaning they still take up space (and cost) until you empty it.

> **Keep this unchecked** to avoid paying for deleted files that remain in the trash.

##### Cost awareness
Persistent Disks are fast but cost more than Cloud Storage.  
Typical rates:  
- **Balanced PD:** ~$0.10–$0.12 / GB / month  
- **SSD PD:** ~$0.17–$0.20 / GB / month  
- **Cloud Storage (Standard):** ~$0.026 / GB / month  

> **Rule of thumb:** use PDs only for active work; store everything else in Cloud Storage.  
> Example: a 200 GB dataset costs **~$24/month on a PD** but only **~$5/month in Cloud Storage**.

Check the latest pricing here:  
- [Persistent Disk & Image pricing](https://cloud.google.com/compute/disks-image-pricing)  
- [Cloud Storage pricing](https://cloud.google.com/storage/pricing)


#### Advanced settings: Networking - Remove External IP Access

- *Don't* **Assign External IP address**: Uncheck this option
  - This ensures your instance is only accessible through secure internal channels rather than the open internet.  
  - Removing the external IP reduces your attack surface and aligns with campus cybersecurity guidance.  

![Remove External IP](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/new-instance-networking.jpg){alt="Remove External IP"}

> **Note:** Managed Workbench instances do not allow you to modify network settings after creation.  Be sure to complete this step or you may need to delete the instance and recreate one from scratch.

### Create notebook 
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


### Load pre-filled Jupyter notebooks
Once your newly created *instance* shows as `Active` (green checkmark), click **Open JupyterLab** to open the instance in Jupyter Lab. From there, we can create as many Jupyter notebooks as we would like within the instance environment. 

We will then select the standard python3 environment to start our first .ipynb notebook (Jupyter notebook). We can use this environment since we aren't doing any training/tuning just yet.

Within the Jupyter notebook, run the following command to clone the lesson repo into our Jupyter environment:

```sh
!git clone https://github.com/qualiaMachine/Intro_GCP_for_ML.git
```

Then, navigate to `/Intro_GCP_for_ML/notebooks/Accessing-and-managing-data.ipynb` to begin the first notebook.


::::::::::::::::::::::::::::::::::::: keypoints 

- Use a small Workbench Instance notebook as a controller to manage larger, resource-intensive tasks.  
- Always navigate to the "Instances" tab in Workbench, since older notebook types are deprecated.  
- Choose the same region for your Workbench Instance and storage bucket to avoid extra transfer costs.  
- Submit training and tuning jobs to scalable instances using the Vertex AI SDK.  
- Labels help track costs effectively, especially in shared or multi-project environments.  
- Workbench Instances come with JupyterLab 3 and GPU frameworks preinstalled, making them an easy entry point for ML workflows.  
- Enable idle auto-stop to avoid unexpected charges when notebooks are left running.  

::::::::::::::::::::::::::::::::::::::::::::::::

