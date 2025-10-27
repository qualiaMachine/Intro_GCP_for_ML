---
title: "Training Models in Vertex AI: Intro"
teaching: 20
exercises: 10
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
So we all can reference helper functions consistently, change directory to your Jupyter home directory.  

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

print("Project:", PROJECT_ID)
```

- `aiplatform.init()`: Sets defaults for project, region, and staging bucket.  

```python
from google.cloud import aiplatform

# Initialize Vertex AI client
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")
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
Before scaling training jobs onto managed resources, it's essential to test your training script locally. This prevents wasting GPU/TPU time on bugs or misconfigured code.  

### Guidelines for testing ML pipelines before scaling
- **Run tests locally first** with small datasets.  
- **Use a subset of your dataset** (1–5%) for fast checks.  
- **Start with minimal compute** before moving to larger accelerators.  
- **Log key metrics** such as loss curves and runtimes.  
- **Verify correctness first** before scaling up.  

::::::::::::::::::::::::::::::::::::::: discussion

### What tests should we do before scaling?  

Before scaling to multiple or more powerful instances (e.g., GPUs or TPUs), it's important to run a few sanity checks. **In your group, discuss:**  

- Which checks do you think are most critical before scaling up?  
- What potential issues might we miss if we skip this step?  

:::::::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::::: solution

- **Data loads correctly** – dataset loads without errors, expected columns exist, missing values handled.  
- **Overfitting check** – train on a tiny dataset (e.g., 100 rows). If it doesn't overfit, something is off.  
- **Loss behavior** – verify training loss decreases and doesn't diverge.  
- **Runtime estimate** – get a rough sense of training time on small data.  
- **Memory estimate** – check approximate memory use.  
- **Save & reload** – ensure model saves, reloads, and infers without errors.  

Skipping these can lead to: silent data bugs, runtime blowups at scale, inefficient experiments, or broken model artifacts.  

:::::::::::::::::::::::::::::::::::::::::::::::::::::

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

Training on this small dataset should take <1 minute. Log runtime as a baseline.  You should see the following output files:

- xgboost-model.joblib  # Python-serialized XGBoost model (Booster) via joblib; load with joblib.load for reuse.
- eval_history.csv      # Per-iteration validation metrics; columns: iter,val_logloss (good for plotting learning curves).
- training.log          # Full stdout/stderr from the run (params, dataset sizes, timings, warnings/errors) for audit/debug.
- metrics.json          # Structured summary: final_val_logloss, num_boost_round, params, train_rows/val_rows, features[], model_uri.


## Training via Vertex AI custom training job
Unlike "local" training, this launches a **managed training job** that runs on scalable compute. Vertex AI handles provisioning, scaling, logging, and saving outputs to GCS.  

### Which machine type to start with?
Start with a small CPU machine like `n1-standard-4`. Only scale up to GPUs/TPUs once you've verified your script. See [Instances for ML on GCP](../instances-for-ML.html) for guidance.  

### Creating a custom training job with the SDK

```python
from google.cloud import aiplatform
import datetime as dt

PROJECT = "doit-rci-mlm25-4626"
REGION = "us-central1"
BUCKET = BUCKET_NAME  # e.g., "endemann_titanic" (same region as REGION)

RUN_ID = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_URI = f"gs://{BUCKET}/artifacts/xgb/{RUN_ID}/model.joblib"  # everything will live beside this

# Staging bucket is only for the SDK's temp code tarball (aiplatform-*.tar.gz)
aiplatform.init(project=PROJECT, location=REGION, staging_bucket=f"gs://{BUCKET}")

job = aiplatform.CustomTrainingJob(
    display_name=f"endemann_xgb_{RUN_ID}",
    script_path="Intro_GCP_VertexAI/code/train_xgboost.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest",
    requirements=["gcsfs"],  # script writes gs://MODEL_URI and sidecar files
)

job.run(
    args=[
        f"--train=gs://{BUCKET}/titanic_train.csv",
        f"--model_out={MODEL_URI}",      # model, metrics.json, eval_history.csv, training.log all go here
        "--max_depth=3",
        "--eta=0.1",
        "--subsample=0.8",
        "--colsample_bytree=0.8",
        "--num_round=100",
    ],
    replica_count=1,
    machine_type="n1-standard-4",
    sync=True,
)

print("Model + logs folder:", MODEL_URI.rsplit("/", 1)[0])

```

This launches a managed training job with Vertex AI. 

## Monitoring training jobs in the Console
1. Go to the Google Cloud Console.  
2. Navigate to **Vertex AI > Training > Custom Jobs**.  
3. Click on your job name to see status, logs, and output model artifacts.  
4. Cancel jobs from the console if needed (be careful not to stop jobs you don't own in shared projects).

#### Visit "training pipelines" to verify it's running. It may take around 5 minutes to finish.

https://console.cloud.google.com/vertex-ai/training/training-pipelines?hl=en&project=doit-rci-mlm25-4626

Should output the following files:

- endemann_titanic/artifacts/xgb/20250924-154740/xgboost-model.joblib  # Python-serialized XGBoost model (Booster) via joblib; load with joblib.load for reuse.
- endemann_titanic/artifacts/xgb/20250924-154740/eval_history.csv      # Per-iteration validation metrics; columns: iter,val_logloss (good for plotting learning curves).
- endemann_titanic/artifacts/xgb/20250924-154740/training.log          # Full stdout/stderr from the run (params, dataset sizes, timings, warnings/errors) for audit/debug.
- endemann_titanic/artifacts/xgb/20250924-154740/metrics.json          # Structured summary: final_val_logloss, num_boost_round, params, train_rows/val_rows, features[], model_uri.

## When training takes too long

Two main options in Vertex AI:  

- **Option 1: Upgrade to more powerful machine types** (e.g., add GPUs like T4, V100, A100).  
- **Option 2: Use distributed training with multiple replicas**.  

### Option 1: Upgrade machine type (preferred first step)
- Works best for small/medium datasets (<10 GB).  
- Avoids the coordination overhead of distributed training.  
- GPUs/TPUs accelerate deep learning tasks significantly.  

### Option 2: Distributed training with multiple replicas
- Supported in Vertex AI for many frameworks.  
- Split data across replicas, each trains a portion, gradients synchronized.  
- More beneficial for very large datasets and long-running jobs.  

### When distributed training makes sense
- Dataset >10–50 GB.  
- Training time >10 hours on single machine.  
- Deep learning workloads that naturally parallelize across GPUs/TPUs.  

::::::::::::::::::::::::::::::::::::: keypoints

- **Environment initialization**: Use `aiplatform.init()` to set defaults for project, region, and bucket.  
- **Local vs managed training**: Test locally before scaling into managed jobs.  
- **Custom jobs**: Vertex AI lets you run scripts as managed training jobs using pre-built or custom containers.  
- **Scaling**: Start small, then scale up to GPUs or distributed jobs as dataset/model size grows.  
- **Monitoring**: Track job logs and artifacts in the Vertex AI Console.  

::::::::::::::::::::::::::::::::::::::::::::::::
