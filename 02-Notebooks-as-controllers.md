---
title: "Notebooks as Controllers"
teaching: 20
exercises: 15
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do you set up and use Vertex AI Workbench notebooks for machine learning tasks?  
- How can you manage compute resources efficiently using a "controller" notebook approach in GCP?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Describe how Vertex AI Workbench notebooks fit into ML/AI workflows on GCP.
- Set up a Jupyter-based Workbench Instance as a lightweight controller to manage compute tasks.
- Configure a Workbench Instance with appropriate machine type, labels, and idle shutdown for cost-efficient orchestration.

::::::::::::::::::::::::::::::::::::::::::::::::

## Setting up our notebook environment
Google Cloud Workbench provides JupyterLab-based environments that can be used to orchestrate ML/AI workflows. In this workshop, we will use a **Workbench Instance**—the recommended option going forward, as other Workbench environments are being deprecated.  

> Workbench Instances come with JupyterLab 3 pre-installed and are configured with GPU-enabled ML frameworks (TensorFlow, PyTorch, etc.), making it easy to start experimenting without additional setup. Learn more in the [Workbench Instances documentation](https://cloud.google.com/vertex-ai/docs/workbench/instances/introduction).  

## Using the notebook as a controller
The notebook instance functions as a *controller* to manage more resource-intensive tasks. By selecting a modest machine type (e.g., `n2-standard-2`), you can perform lightweight operations locally in the notebook while using the **Vertex AI Python SDK** to launch compute-heavy jobs on larger machines (e.g., GPU-accelerated) when needed.

This approach minimizes costs while giving you access to scalable infrastructure for demanding tasks like model training, batch prediction, and hyperparameter tuning.

One practical advantage of Workbench notebooks: **authentication is automatic.** A Workbench VM inherits the permissions of its attached service account, so calls to Cloud Storage, Vertex AI, and the Gemini API work with no extra credential setup — no API keys or login commands needed. If you later run the same code from your laptop or an HPC cluster, you'll need to set up credentials separately (see the [GCP authentication docs](https://cloud.google.com/docs/authentication)). (Prefer working from a terminal? [Episode 8: CLI Workflows](08-CLI-workflows.md) covers how to do everything in this workshop using `gcloud` commands instead of notebooks.)

We will follow these steps to create our first Workbench Instance:

### 1. Navigate to Workbench

- Open the **Google Cloud Console** ([console.cloud.google.com](https://console.cloud.google.com/)) — this is the web dashboard where you manage all GCP resources. Search for "Workbench."
- Click the "Instances" tab (this is the supported path going forward).

### 2. Create a new Workbench Instance

#### Initial settings

- Click **Create New** near the top of the Workbench page
- **Name**: Use the convention `lastname-purpose` (e.g., `doe-workshop`). We'll use a single instance for training, tuning, RAG, and more, so `workshop` is a good general-purpose label.
- **Region**: Select `us-central1`. When we create a storage bucket in [Episode 3](03-Data-storage-and-access.md), we'll use the same region — keeping compute and storage co-located avoids cross-region transfer charges and keeps data access fast.
- **Zone:** `us-central1-a` (or another zone in `us-central1`, like `-b` or `-c`)  
  - If capacity or GPU availability is limited in one zone, switch to another zone in the same region.
- **NVIDIA T4 GPU:** Leave unchecked for now  
  - We will request GPUs for training jobs separately. Attaching here increases idle costs.
- **Apache Spark and BigQuery Kernels:** Leave unchecked
  - BigQuery kernels let you run SQL analytics directly in a notebook, but we won't need them in this workshop. Leave unchecked to avoid pulling extra container images.
- **Network in this project:** If you're working in a shared workshop environment, select the network provided by your administrator (shared environments typically do not allow using external or default networks). If using a personal GCP project, the default network is fine.
- **Network / Subnetwork:** Leave as pre-filled.
![Notebook settings (part 1)](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/new-instance-settings1.jpg){alt="Notebook settings (part1)"}

#### Advanced settings: Details (tagging)

- **IMPORTANT:** Open the "Advanced options" menu next.
  -  **Labels (required for cost tracking):**  Under the Details menu, add the following tags (all lowercase) so that you can track the total cost of your activity on GCP later:
      - `name = lastname` (your last name)
      - `purpose = workshop`
        
![Required tags for notebook.](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/new-instance-tags.jpg){alt="Screenshot showing required tags for notebook"}

#### Advanced Settings: Environment

Leave environment settings at their defaults for this workshop. Workbench uses JupyterLab 3 by default with NVIDIA GPU drivers, CUDA, and common ML frameworks preinstalled. For future reference, you can optionally select JupyterLab 4, provide a custom Docker image, or specify a post-startup script (`gs://path/to/script.sh`) to auto-configure the instance at boot.

#### Advanced settings: Machine Type 

- **Machine type**: Select a small machine (e.g., `n2-standard-2`, ~$0.07/hr) to act as the controller.
  - This keeps costs low while you delegate heavy lifting to training jobs.
  - For guidance on common machine types and their costs, see [Compute for ML](../compute-for-ML.html). For help deciding when you need cloud hardware at all, see [When does model size justify cloud compute?](01-Introduction.md#when-does-model-size-justify-cloud-compute) in Episode 1.

- **Set idle shutdown**: To save on costs when you aren't doing anything in your notebook, lower the default idle shutdown time to **60 (minutes)**.

![Enable Idle Shutdown](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/new-instance-idleshutdown.jpg){alt="Set Idle Shutdown"}

#### Advanced Settings: Disks

Leave disk settings at their defaults for this workshop. Each Workbench Instance has two disks: a **boot disk** (100 GB — holds the OS and libraries) and a **data disk** (150 GB default — holds your datasets and outputs). Both use Balanced Persistent Disks. Keep "Delete to trash" unchecked so deleted files free space immediately.

**Rule of thumb:** allocate ≈ 2× your dataset size for the data disk, and keep bulk data in Cloud Storage (`gs://`) rather than on local disk — PDs cost ~$0.10/GB/month vs. ~$0.02/GB/month for Cloud Storage.

::::::::::::::::::::::::::::::::::::: callout

#### Disk sizing and cost details

- **Boot disk:** Rarely needs resizing. Increase to 150–200 GB only for large custom environments or multiple frameworks.
- **Data disk:** Use SSD PD only for high-I/O workloads. Disks can be resized anytime without downtime, so start small and expand when needed.
- **Cost comparison:** A 200 GB dataset costs ~$24/month on a PD but only ~$5/month in Cloud Storage.
- **Pricing:** [Persistent Disk pricing](https://cloud.google.com/compute/disks-image-pricing) · [Cloud Storage pricing](https://cloud.google.com/storage/pricing)

::::::::::::::::::::::::::::::::::::::::::::::::


#### Advanced settings: Networking - External IP Access

- **Assign External IP address**: Leave this option checked — you need an external IP.  

### Create notebook

- Click **Create** to create the instance. Provisioning typically takes 3–5 minutes. You'll see the status change from "Provisioning" to "Active" with a green checkmark. While waiting, work through the challenges below.

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Notebook Roles

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

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Controller Cost Estimate

Your controller notebook uses an `n2-standard-2` instance (~$0.07/hr — see [Compute for ML](../compute-for-ML.html) for other common machine types and costs).

- Estimate the monthly cost if you use it 8 hours/day, 5 days/week, with idle shutdown enabled.
- Compare that to leaving it running 24/7 for the same month.

:::::::::::::::: solution

- **With idle shutdown:** 8 hrs × 5 days × 4 weeks = 160 hrs → 160 × $0.07 ≈ **$11.20/month**
- **Running 24/7:** 24 hrs × 30 days = 720 hrs → 720 × $0.07 ≈ **$50.40/month**
- Idle shutdown saves you ~$39/month on a single small controller instance. The savings are even larger for bigger machine types.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

### Managing your instance

You don't have to wait for idle shutdown — you can **manually stop** your instance anytime from the Workbench Instances list by selecting the checkbox and clicking **Stop**. To resume work, click **Start**. You only pay for compute while the instance is running (disk charges continue while stopped).

To permanently remove an instance, select it and click **Delete**. Full cleanup is covered in Episode 9.

### Managing training and tuning with the controller notebook
In the following episodes, we will use the **Vertex AI Python SDK (`google-cloud-aiplatform`)** from this notebook to submit compute-heavy tasks on more powerful machines. Examples include:

- Training a model on a GPU-backed instance.
- Running hyperparameter tuning jobs managed by Vertex AI.

Here's how the notebook, jobs, and storage connect:

![Training and tuning workflow](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/diagram1_training_and_tuning.svg){alt="Architecture diagram showing how a lightweight Workbench notebook uses the Vertex AI SDK to launch training jobs and HP tuning jobs on powerful GPUs, with all artifacts stored in GCS."}

This pattern keeps costs low by running your notebook on a modest VM while only incurring charges for larger resources when they are actively in use.

::::::::::::::::::::::::::::::::::::: callout

#### You don't need a notebook to use Vertex AI

We use **Vertex AI Workbench** notebooks rather than a plain local JupyterLab because Workbench comes pre-configured with ML frameworks, GPU drivers, and — most importantly — automatic GCP authentication through the VM's service account. You could run a self-managed Jupyter server on a Compute Engine VM, but you'd have to install libraries and configure credentials yourself. Working through Workbench also gets you comfortable with the GCP Console — navigating services, checking instance status, reading logs — skills that transfer to every other GCP workflow.

That said, **notebooks are not required** for any of the workflows covered here. Everything we do through the Python SDK (submitting training jobs, running hyperparameter tuning, calling the Gemini API) can also be done from:

- A **plain Python script** run from your terminal or an HPC scheduler.
- The **`gcloud` CLI** (e.g., `gcloud ai custom-jobs create ...`) for submitting and managing jobs directly from the command line.
- A **CI/CD pipeline** (GitHub Actions, Cloud Build, etc.) that triggers training runs automatically.

The real work happens in the training scripts and SDK calls — the notebook is just a convenient wrapper for orchestrating them.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: callout

#### Troubleshooting

- **VM stuck in "Provisioning" for more than 5 minutes?** Try deleting the instance and re-creating it in a different zone within the same region (e.g., `us-central1-b` instead of `us-central1-a`).
- **Instance stopped unexpectedly?** Check the idle shutdown setting — it may have timed out. Restart from the Instances list by clicking **Start**.
- **Can't see the project or get permission errors?** Ensure you're signed into the correct Google account and that IAM permissions have propagated (this can take a few minutes after initial setup).

::::::::::::::::::::::::::::::::::::::::::::::::

### Load pre-filled Jupyter notebooks
Once your instance shows as "Active" (green checkmark), click **Open JupyterLab**. From the Launcher, select **Python 3 (ipykernel)** under Notebook to create a new notebook — we don't need the TensorFlow or PyTorch kernels yet, as those are used in later episodes for training jobs.

Run the following command to clone the lesson repository. This contains pre-filled notebooks for each episode and the training scripts we'll use later, so you won't need to write boilerplate code from scratch.

```sh
!git clone https://github.com/qualiaMachine/Intro_GCP_for_ML.git
```

Then, navigate to `/Intro_GCP_for_ML/notebooks/03-Data-storage-and-access.ipynb` to begin the next episode.

::::::::::::::::::::::::::::::::::::: keypoints

- Use a small Workbench Instance as a controller — delegate heavy training to Vertex AI jobs.
- Workbench VMs inherit service account permissions automatically, simplifying authentication.
- Choose the same region for your Workbench Instance and storage bucket to avoid extra transfer costs.
- Apply labels to all resources for cost tracking, and enable idle auto-stop to avoid surprise charges.

::::::::::::::::::::::::::::::::::::::::::::::::

