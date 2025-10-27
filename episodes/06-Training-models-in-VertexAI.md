---
title: "Training Models in Vertex AI: Intro"
teaching: 30
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- What are the differences between training locally in a Vertex AI notebook and using Vertex AI-managed training jobs?  
- How do custom training jobs in Vertex AI streamline the training process for various frameworks?  
- How does Vertex AI handle scaling across CPUs, GPUs, and TPUs?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand the difference between local training in a Vertex AI Workbench notebook and submitting managed training jobs.  
- Learn to configure and use Vertex AI custom training jobs for different frameworks (e.g., XGBoost, PyTorch, SKLearn).  
- Understand scaling options in Vertex AI, including when to use CPUs, GPUs, or TPUs.  
- Compare performance, cost, and setup between custom scripts and pre-built containers in Vertex AI.  
- Conduct training with data stored in GCS and monitor training job status using the Google Cloud Console.  

::::::::::::::::::::::::::::::::::::::::::::::::

## Initial setup 

#### 1. Open a new .ipynb notebook
Navigate to `/Intro_GCP_for_ML/notebooks/06-Training-models-in-VertexAI.ipynb` to begin this notebook.

#### 2. CD to instance home directory
To ensure we're all in the saming starting spot, change directory to your Jupyter home directory.

```python
%cd /home/jupyter/
```

#### 3. Initialize Vertex AI environment
This code initializes the Vertex AI environment by importing the Python SDK, setting the project, region, and defining a GCS bucket for input/output data.

- `PROJECT_ID`: Identifies your GCP project.  
- `REGION`: Determines where training jobs run (choose a region close to your data).  
- `staging_bucket`: A GCS bucket for storing datasets, model artifacts, and job outputs.  
```python
from google.cloud import storage
client = storage.Client()
PROJECT_ID = client.project
REGION = "us-central1"
BUCKET_NAME = "sinkorswim-johndoe-titanic" # ADJUST to your bucket's name

print(f"project = {PROJECT_ID}\nregion = {REGION}\nbucket = {BUCKET_NAME}")
```


#### 4. Get code from GitHub repo (skip if already completed)
If you didn't complete earlier episodes, clone our code repo before moving forward. Check to make sure we're in our Jupyter home folder first.  

```python
#%cd /home/jupyter/
```

```python
#!git clone https://github.com/qualiaMachine/Intro_GCP_for_ML.git
```

## Testing train.py locally in the notebook

::::::::::::::::::::::::::::::::::::::: challenge

### Understanding the XGBoost Training Script (GCP version)

Take a moment to review the `train_xgboost.py` script we're using on GCP found in `Intro_GCP-for_ML/scripts/train_xgboost.py`. This script handles preprocessing, training, and saving an XGBoost model, while supporting **local paths** and **GCS (`gs://`) paths**, and it adapts to **Vertex AI** conventions (e.g., `AIP_MODEL_DIR`).

Try answering the following questions:

1. **Data preprocessing**: What transformations are applied to the dataset before training?

2. **Training function**: What does the `train_model()` function do? Why print the training time?

3. **Command-line arguments**: What is the purpose of `argparse` in this script? How would you change the number of training rounds?

4. **Handling local vs. GCP runs**: How does the script let you run the same code locally, in Workbench, or as a Vertex AI job? Which environment variable controls where the model artifact is written?

5. **Training and saving the model**: What format is the dataset converted to before training, and why? How does the script save to a local path vs. a `gs://` destination?

After reviewing, discuss any questions or observations with your group.

:::::::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::::: solution

### Solution

1. **Data preprocessing**: The script fills missing values (`Age` with median, `Embarked` with mode), maps categorical fields to numeric (`Sex` → {male:1, female:0}, `Embarked` → {S:0, C:1, Q:2}), and drops non-predictive columns (`Name`, `Ticket`, `Cabin`).

2. **Training function**: `train_model()` constructs and fits an XGBoost model with the provided parameters and prints wall-clock training time. Timing helps compare runs and make sensible scaling choices.

3. **Command-line arguments**: `argparse` lets you set hyperparameters and file paths without editing code (e.g., `--max_depth`, `--eta`, `--num_round`, `--train`). To change rounds:  `python train_xgboost.py --num_round 200`

4. **Handling local vs. GCP runs**:  
   - **Input**: You pass `--train` as either a local path (`train.csv`) or a GCS URI (`gs://bucket/path.csv`). The script automatically detects `gs://` and reads the file directly from Cloud Storage using the Python client.  
   - **Output**: If the environment variable `AIP_MODEL_DIR` is set (as it is in Vertex AI CustomJobs), the trained model is written there—often a `gs://` path. Otherwise, the model is saved in the current working directory, which works seamlessly in both local and Workbench environments.

5. **Training and saving the model**:  
   The training data is converted into an **XGBoost `DMatrix`**, an optimized format that speeds up training and reduces memory use. The trained model is serialized with `joblib`. When saving locally, the file is written directly to disk. If saving to a Cloud Storage path (`gs://...`), the model is first saved to a temporary file and then uploaded to the specified bucket.

:::::::::::::::::::::::::::::::::::::::


Before scaling training jobs onto managed resources, it's essential to test your training script locally. This prevents wasting GPU/TPU time on bugs or misconfigured code.  

### Guidelines for testing ML pipelines before scaling

- **Run tests locally first** with small datasets.  
- **Use a subset of your dataset** (1–5%) for fast checks.  
- **Start with minimal compute** before moving to larger accelerators.  
- **Log key metrics** such as loss curves and runtimes.  
- **Verify correctness first** before scaling up.  


### What tests should we do before scaling?  

Before scaling to multiple or more powerful instances (e.g., GPUs or TPUs), it's important to run a few sanity checks. Skipping these can lead to: silent data bugs, runtime blowups at scale, inefficient experiments, or broken model artifacts.  

Here is a non-exhaustive list of suggested tests to perform before scaling up your compute needs.

- **Data loads correctly** – dataset loads without errors, expected columns exist, missing values handled.  
- **Overfitting check** – train on a tiny dataset (e.g., 100 rows). If it doesn't overfit, something is off.  
- **Loss behavior** – verify training loss decreases and doesn't diverge.  
- **Runtime estimate** – get a rough sense of training time on small data.  
- **Memory estimate** – check approximate memory use.  
- **Save & reload** – ensure model saves, reloads, and infers without errors.  


## Download data into notebook environment
Sometimes it's helpful to keep a copy of data in your notebook VM for quick iteration, even though **GCS is the preferred storage location**.  

```python
from google.cloud import storage

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

blob = bucket.blob("titanic_train.csv")
blob.download_to_filename("titanic_train.csv")

print("Downloaded titanic_train.csv")
```

## Local test run of train.py

```python
import time as t

start = t.time()

# Example: run your custom training script with args
!python Intro_GCP_for_ML/scripts/train_xgboost.py --max_depth 3 --eta 0.1 --subsample 0.8 --colsample_bytree 0.8 --num_round 100 --train titanic_train.csv

print(f"Total local runtime: {t.time() - start:.2f} seconds")
```

Training on this small dataset should take <1 minute. Log runtime as a baseline.  You should see the following output file:

- `xgboost-model`  # Python-serialized XGBoost model (Booster) via joblib; load with joblib.load for reuse.

## Training via Vertex AI custom training job
Unlike "local" training using our notebook's VM, this next approach launches a **managed training job** that runs on scalable compute. Vertex AI handles provisioning, scaling, logging, and saving outputs to GCS.  

### Which machine type to start with?
Start with a small CPU machine like `n1-standard-4`. Only scale up to GPUs/TPUs once you've verified your script. See [Instances for ML on GCP](https://qualiamachine.github.io/Intro_GCP_for_ML/instances-for-ML.html) for guidance.  

```python
MACHINE = 'n1-standard-4'
```

### Creating a custom training job with the SDK

We'll first initialize the Vertex AI platform with our environment variables. We'll also set a `RUN_ID` and `ARTIFACT_DIR` to help store outputs. 

```python
from google.cloud import aiplatform
import datetime as dt
RUN_ID = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"project = {PROJECT_ID}\nregion = {REGION}\nbucket = {BUCKET_NAME}")
ARTIFACT_DIR = f"gs://{BUCKET_NAME}/artifacts/xgb/{RUN_ID}/"  # everything will live beside this

# Staging bucket is only for the SDK's temp code tarball (aiplatform-*.tar.gz)
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}/.vertex_staging")
```

This next section defines a custom training job in Vertex AI, specifying how and where the training code will run.  It points to your training script (`train_xgboost.py`), uses Google's prebuilt XGBoost training container image, and installs any extra dependencies your script needs (in this case, `google-cloud-storage` for accessing GCS).  The `display_name` sets a readable name for tracking the job in the Vertex AI console.

```python

job = aiplatform.CustomTrainingJob(
    display_name=f"endemann_xgb_{RUN_ID}",
    script_path="Intro_GCP_for_ML/scripts/train_xgboost.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest",
    requirements=["google-cloud-storage"],  # your script uses storage.Client()
)
```

Finally, this next block launches the custom training job on Vertex AI using the configuration defined earlier.  
The `args` list passes command-line parameters directly into your training script, including hyperparameters and the path to the training data in GCS.  
`base_output_dir` specifies where all outputs (model, metrics, logs) will be written in Cloud Storage, and `machine_type` controls the compute resources used for training.  
When `sync=True`, the notebook waits until the job finishes before continuing, making it easier to inspect results immediately after training.

```python
job.run(
    args=[
        f"--train=gs://{BUCKET_NAME}/titanic_train.csv",
        "--max_depth=3",
        "--eta=0.1",
        "--subsample=0.8",
        "--colsample_bytree=0.8",
        "--num_round=100",
    ],
    replica_count=1,
    machine_type=MACHINE, # MACHINE variable defined above; adjust to something more powerful when needed
    base_output_dir=ARTIFACT_DIR,  # sets AIP_MODEL_DIR for your script
    sync=True,
)

print("Model + logs folder:", ARTIFACT_DIR)

```

This launches a managed training job with Vertex AI. It should take 2-5 minutes for the training job to complete. 

### Understanding the training output message

After your job finishes, you may see a message like: `Training did not produce a Managed Model returning None.` This is expected when running a `CustomTrainingJob` without specifying deployment parameters.  Vertex AI supports two modes:

- **CustomTrainingJob (research/development)** – You control training and save models/logs to Cloud Storage via `AIP_MODEL_DIR`. This is ideal for experimentation and cost control.
- **TrainingPipeline (for deployment)** – You include `model_serving_container_image_uri` and `model_display_name`, and Vertex automatically registers a *Managed Model* in the Model Registry for deployment to an endpoint.

In our setup, we're intentionally using the simpler **CustomTrainingJob** path. Your trained model is safely stored under your specified artifact directory (e.g., `gs://{BUCKET_NAME}/artifacts/xgb/{RUN_ID}/`), and you can later register or deploy it manually when ready.


## Monitoring training jobs in the Console
1. Go to the Google Cloud Console.  
2. Navigate to **Vertex AI > Training > Custom Jobs**.  
3. Click on your job name to see status, logs, and output model artifacts.  
4. Cancel jobs from the console if needed (be careful not to stop jobs you don't own in shared projects).

#### Visit "training pipelines" to verify it's running.

https://console.cloud.google.com/vertex-ai/training/training-pipelines?hl=en&project=doit-rci-mlm25-4626

## Training artifacts

After the training run completes, we can manually view our bucket using the Google Cloud Console or run the below code.

```python
total_size_bytes = 0
# bucket = client.bucket(BUCKET_NAME)

for blob in client.list_blobs(BUCKET_NAME):
    total_size_bytes += blob.size
    print(blob.name)

total_size_mb = total_size_bytes / (1024**2)
print(f"Total size of bucket '{BUCKET_NAME}': {total_size_mb:.2f} MB")
```

#### Training Artifacts  →  `gs://<bucket>/artifacts/<run_id>/`
This is your *intended output location*, set via `base_output_dir`.  
It contains everything your training script explicitly writes. In our case, this includes:

- **`{BUCKET_NAME}/artifacts/xgb/{RUN_ID}/xgboost-model`** — Serialized XGBoost model (Booster) saved via `joblib`; reload later with `joblib.load()` for reuse or deployment.  


#### System-Generated Files
Additional system-generated files (e.g., Vertex's `.tar.gz` code package or `executor_output.json`) will appear under `.vertex_staging/` and can be safely ignored or auto-deleted via lifecycle rules.

### When training takes too long  

Two main options in Vertex AI:  

**Option 1: Upgrade to more powerful machine types**  
- The simplest way to reduce training time is to use a larger machine or add GPUs (e.g., T4, V100, A100).  
- This works best for small or medium datasets (<10 GB) and avoids the coordination overhead of distributed training.  
- GPUs and TPUs can accelerate deep learning workloads significantly.  

**Option 2: Use distributed training with multiple replicas**  
- Vertex AI supports distributed training for many frameworks.  
- The dataset is split across replicas, each training a portion of the data with synchronized gradient updates.  
- This approach is most useful for large datasets and long-running jobs.  

**When distributed training makes sense**  
- Dataset size exceeds 10–50 GB.  
- Training on a single machine takes more than 10 hours.  
- The model is a deep learning workload that scales naturally across GPUs or TPUs.  

We will explore both options more in depth in the next episode when we train a neural network.

::::::::::::::::::::::::::::::::::::: keypoints

- **Environment initialization**: Use `aiplatform.init()` to set defaults for project, region, and bucket.  
- **Local vs managed training**: Test locally before scaling into managed jobs.  
- **Custom jobs**: Vertex AI lets you run scripts as managed training jobs using pre-built or custom containers.  
- **Scaling**: Start small, then scale up to GPUs or distributed jobs as dataset/model size grows.  
- **Monitoring**: Track job logs and artifacts in the Vertex AI Console.  

::::::::::::::::::::::::::::::::::::::::::::::::
