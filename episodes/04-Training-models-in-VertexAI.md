---
title: "Training Models in Vertex AI: Intro"
teaching: 25
exercises: 15
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

::::::::::::::::::::::::::::::::::::: callout

### Cost awareness: training jobs

Training jobs bill per VM-hour while the job is running. An `n1-standard-4` (CPU) costs ~`$0.19`/hr; adding a T4 GPU brings the total to ~`$0.54`/hr. Jobs automatically stop (and stop billing) when the script finishes. For a complete cost reference, see the [Compute for ML](../compute-for-ML.html) page and the cost table in [Episode 9](09-Resource-management-cleanup.md).

::::::::::::::::::::::::::::::::::::::::::::::::

## Initial setup

#### 1. Open pre-filled notebook
Navigate to `/Intro_GCP_for_ML/notebooks/04-Training-models-in-VertexAI.ipynb` to begin this notebook.

#### 2. CD to instance home directory
To ensure we're all in the same starting spot, change directory to your Jupyter home directory.

```python
%cd /home/jupyter/
```

#### 3. Set environment variables 
This code initializes the Vertex AI environment by importing the Python SDK, setting the project, region, and defining a GCS bucket for input/output data.

- `PROJECT_ID`: Identifies your GCP project.  
- `REGION`: Determines where training jobs run (choose a region close to your data).  

```python
from google.cloud import storage
client = storage.Client()
PROJECT_ID = client.project
REGION = "us-central1"
BUCKET_NAME = "johndoe-titanic" # ADJUST to your bucket's name
LAST_NAME = "DOE" # ADJUST to your last name or name
print(f"project = {PROJECT_ID}\nregion = {REGION}\nbucket = {BUCKET_NAME}")
```

## Testing train_xgboost.py locally in the notebook

Before submitting a managed training job to Vertex AI, let's first examine and test the training script on our notebook VM. This ensures the code runs without errors before we spend money on cloud compute.

::::::::::::::::::::::::::::::::::::::: challenge

### Understanding the XGBoost Training Script (GCP version)

Take a moment to review the `train_xgboost.py` script we're using on GCP found in `Intro_GCP_for_ML/scripts/train_xgboost.py`. This script handles preprocessing, training, and saving an XGBoost model, while supporting local paths and GCS (`gs://`) paths, and it adapts to Vertex AI conventions (e.g., `AIP_MODEL_DIR`).

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

1. **Data preprocessing**: The script fills missing values (`Age` with median, `Embarked` with mode), maps categorical fields to numeric (`Sex` → {male:1, female:0}, `Embarked` → {S:0, C:1, Q:2}), and drops non-predictive columns (`PassengerId`, `Name`, `Ticket`, `Cabin`).
2. **Training function**: `train_model()` constructs and fits an XGBoost model with the provided parameters and prints wall-clock training time. Timing helps compare runs and make sensible scaling choices.
3. **Command-line arguments**: `argparse` lets you set hyperparameters and file paths without editing code (e.g., `--max_depth`, `--eta`, `--num_round`, `--train`). To change rounds:  `python train_xgboost.py --num_round 200`
4. **Handling local vs. GCP runs**:  
   - **Input**: You pass `--train` as either a local path (`train.csv`) or a GCS URI (`gs://bucket/path.csv`). The script automatically detects `gs://` and reads the file directly from Cloud Storage using the Python client.  
   - **Output**: If the environment variable `AIP_MODEL_DIR` is set (as it is in Vertex AI CustomJobs), the trained model is written there—often a `gs://` path. Otherwise, the model is saved in the current working directory, which works seamlessly in both local and Workbench environments.
5. **Training and saving the model**:  
   The training data is converted into an **XGBoost `DMatrix`**, an optimized format that speeds up training and reduces memory use. The trained model is serialized with `joblib`. When saving locally, the file is written directly to disk. If saving to a Cloud Storage path (`gs://...`), the model is first saved to a temporary file and then uploaded to the specified bucket.

:::::::::::::::::::::::::::::::::::::::


Before scaling training jobs onto managed resources, it's essential to test your training script locally. This prevents wasting GPU/TPU time on bugs or misconfigured code. Skipping these checks can lead to silent data bugs, runtime blowups at scale, inefficient experiments, or broken model artifacts.

### Sanity checks before scaling

- **Reproducibility** – Do you get the same result each time? If not, set seeds controlling randomness.
- **Data loads correctly** – Dataset loads without errors, expected columns exist, missing values handled.
- **Overfitting check** – Train on a tiny dataset (e.g., 100 rows). If it doesn't overfit, something is off.
- **Loss behavior** – Verify training loss decreases and doesn't diverge.
- **Runtime estimate** – Get a rough sense of training time on small data before committing to large compute.
- **Memory estimate** – Check approximate memory use to choose the right machine type.
- **Save & reload** – Ensure model saves, reloads, and infers without errors.


## Download data into notebook environment
Sometimes it's helpful to keep a copy of data in your notebook VM for quick iteration, even though **GCS is the preferred storage location**. For example, downloading locally lets you test your training script without any GCS dependencies, making debugging faster. Once you've verified everything works, the actual Vertex AI job will read directly from GCS.

```python
bucket = client.bucket(BUCKET_NAME)

blob = bucket.blob("titanic_train.csv")
blob.download_to_filename("/home/jupyter/titanic_train.csv")

print("Downloaded titanic_train.csv")
```

## Local test run of train_xgboost.py

**Outside of this workshop, you should run these kinds of tests on your local laptop or lab PC when possible.** We're using the Workbench VM here only for convenience in this workshop setting, but this does incur a small fee for our running VM. 

- For large datasets, use a small representative sample of the total dataset when testing locally (i.e., just to verify that code is working and model overfits nearly perfectly after training enough epochs)
- For larger models, use smaller model equivalents (e.g., 100M vs 7B params) when testing locally

```python
# We need to add xgboost to our VM before running the script
!pip install xgboost
```

```python
# Training configuration parameters for XGBoost
MAX_DEPTH = 3         # maximum depth of each decision tree (controls model complexity)
ETA = 0.1             # learning rate (how much each tree contributes to the overall model)
SUBSAMPLE = 0.8       # fraction of training samples used per boosting round (prevents overfitting)
COLSAMPLE = 0.8       # fraction of features (columns) sampled per tree (adds randomness and diversity)
NUM_ROUND = 100       # number of boosting iterations (trees) to train

import time as t
start = t.time()

# Run the custom training script with hyperparameters defined above
!python Intro_GCP_for_ML/scripts/train_xgboost.py \
    --max_depth $MAX_DEPTH \
    --eta $ETA \
    --subsample $SUBSAMPLE \
    --colsample_bytree $COLSAMPLE \
    --num_round $NUM_ROUND \
    --train titanic_train.csv

print(f"Total local runtime: {t.time() - start:.2f} seconds")

```

Training on this small dataset should take <1 minute. Log runtime as a baseline. You should see the following output file:

- `xgboost-model` — Serialized XGBoost model (Booster) via joblib; load with `joblib.load()` for reuse.

## Evaluate the trained model on validation data

Now that we've trained and saved an XGBoost model, we want to do the most important sanity check:  
**Does this model make reasonable predictions on unseen data?**

This step:
1. Loads the serialized model artifact that was written by `train_xgboost.py`
2. Loads a test set of Titanic passenger data
3. Applies the same preprocessing as training
4. Generates predictions
5. Computes simple accuracy

First, we'll download the test data

```python
blob = bucket.blob("titanic_test.csv")
blob.download_to_filename("titanic_test.csv")

print("Downloaded titanic_test.csv")
```

Then, we apply the same preprocessing function used by our training script before applying the model to our data.

> **Note:** The `import` below treats the repo as a Python package. This works because we cloned the repo into `/home/jupyter/` and the directory contains an `__init__.py`. If you get an `ImportError`, make sure your working directory is `/home/jupyter/` (run `%cd /home/jupyter/` first).

> **Note on test data:** The training script internally splits its input data 80/20 for training and validation. The `titanic_test.csv` file we use here is a **separate, held-out test set** that was never seen during training — not even by the internal validation split. This gives us an unbiased measure of model performance.

```python
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score
from Intro_GCP_for_ML.scripts.train_xgboost import preprocess_data  # reuse same preprocessing

# Load test data
test_df = pd.read_csv("titanic_test.csv")

# Apply same preprocessing from training
X_test, y_test = preprocess_data(test_df)

# Load trained model from local file
model = joblib.load("xgboost-model")

# Predict on test data
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)

# Compute accuracy
acc = accuracy_score(y_test, y_pred_binary)
print(f"Test accuracy: {acc:.3f}")
```

You should see test accuracy in the range of **0.78–0.82**. If accuracy is significantly lower, double-check that the test data downloaded correctly and that the preprocessing matches the training script.

::::::::::::::::::::::::::::::::::::::: challenge

### Experiment with hyperparameters

Try changing `NUM_ROUND` to `200` and re-running the local training and evaluation cells above. Does accuracy improve? How does the runtime change? Then try `MAX_DEPTH = 6`. What happens to accuracy — does the model improve, or does it start overfitting?

:::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::: solution

### Solution

Increasing `NUM_ROUND` from 100 to 200 may marginally improve accuracy but roughly doubles runtime. Increasing `MAX_DEPTH` from 3 to 6 lets trees capture more complex patterns but can lead to overfitting on a small dataset like Titanic — you may see training accuracy increase while test accuracy stays flat or drops. This is why testing hyperparameters locally before scaling is important.

:::::::::::::::::::::::::::::::::::::::

## Training via Vertex AI custom training job
Unlike "local" training using our notebook's VM, this next approach launches a **managed training job** that runs on scalable compute. Vertex AI handles provisioning, scaling, logging, and saving outputs to GCS.  

### Which machine type to start with?
Start with a small CPU machine like `n1-standard-4`. Only scale up to GPUs/TPUs once you've verified your script. See [Compute for ML](https://qualiamachine.github.io/Intro_GCP_for_ML/compute-for-ML.html) for guidance.  

```python
MACHINE = 'n1-standard-4'
```

### Creating a custom training job with the SDK

> **Reminder:** We're using the Python SDK from a notebook here, but the same `aiplatform.CustomTrainingJob` calls work identically in a standalone `.py` script, a shell session, or a CI pipeline. You can also submit jobs entirely from the command line with `gcloud ai custom-jobs create`. See the callout in Episode 2 for more details.

We'll first initialize the Vertex AI platform with our environment variables. We'll also set a `RUN_ID` and `ARTIFACT_DIR` to help store outputs.

```python
from google.cloud import aiplatform
import datetime as dt
RUN_ID = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
ARTIFACT_DIR = f"gs://{BUCKET_NAME}/artifacts/xgb/{RUN_ID}/"  # everything will live beside this
print(f"project = {PROJECT_ID}\nregion = {REGION}\nbucket = {BUCKET_NAME}\nartifact_dir = {ARTIFACT_DIR}")

# Staging bucket is only for the SDK's temp code tarball (aiplatform-*.tar.gz)
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}/.vertex_staging")
```

This next section defines a custom training job in Vertex AI, specifying how and where the training code will run. It points to your training script (`train_xgboost.py`), uses Google's prebuilt XGBoost training container image (which already includes common dependencies like `google-cloud-storage`), and sets a `display_name` for tracking the job in the Vertex AI console.

> **Tip:** If your script needs packages not included in the prebuilt container, you can pass a `requirements` list to `CustomTrainingJob` (e.g., `requirements=["scikit-learn>=1.3"]`).

#### Prebuilt containers for training
Vertex AI provides prebuilt Docker container images for model training. These containers are organized by machine learning frameworks and framework versions and include common dependencies that you might want to use in your training code. To learn more about prebuilt training containers, see [Prebuilt containers for custom training](https://docs.cloud.google.com/vertex-ai/docs/training/pre-built-containers).

```python

job = aiplatform.CustomTrainingJob(
    display_name=f"{LAST_NAME}_xgb_{RUN_ID}",
    script_path="Intro_GCP_for_ML/scripts/train_xgboost.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest",
)
```

Finally, this next block launches the custom training job on Vertex AI using the configuration defined earlier. **We won't be charged for our selected `MACHINE` until we run the below code using `job.run()`.** For an `n1-standard-4` running 2–5 minutes, expect a cost of roughly **`$0.01`–`$0.02`** — negligible, but good to be aware of as you scale to larger machines. This marks the point when our script actually begins executing remotely on the Vertex training infrastructure. Once `job.run()` is called, Vertex handles packaging your training script, transferring it to the managed training environment, provisioning the requested compute instance, and monitoring the run. The job's status and logs can be viewed directly in the Vertex AI Console under Training → Custom jobs.

If you need to cancel or modify a job mid-run, you can do so from the console or via the SDK by calling job.cancel(). When the job completes, Vertex automatically tears down the compute resources so you only pay for the active training time.

- The `args` list passes command-line parameters directly into your training script, including hyperparameters and the path to the training data in GCS.  
- `base_output_dir` specifies where all outputs (model, metrics, logs) will be written in Cloud Storage
- `machine_type` controls the compute resources used for training.
- When `sync=True`, the notebook waits until the job finishes before continuing, making it easier to inspect results immediately after training.

```python
job.run(
    args=[
        f"--train=gs://{BUCKET_NAME}/titanic_train.csv",
        f"--max_depth={MAX_DEPTH}",
        f"--eta={ETA}",
        f"--subsample={SUBSAMPLE}",
        f"--colsample_bytree={COLSAMPLE}",
        f"--num_round={NUM_ROUND}",
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

#### Visit the console to verify it's running.

Navigate to **Vertex AI > Training > Custom Jobs** in the [Google Cloud Console](https://console.cloud.google.com/vertex-ai/training/custom-jobs) to view your running or completed jobs.

### If your job fails

Job failures are common when first getting started. Here's how to debug:

1. **Check the logs first.** In the Console, click your job name → **Logs** tab. The error message is usually near the bottom.
2. **Common failure modes:**
   - **Quota exceeded** — Your project may not have enough quota for the requested machine type. Check **IAM & Admin > Quotas**.
   - **Script error** — A bug in your training script. The traceback will appear in the logs. Fix the bug and re-run locally before resubmitting.
   - **Wrong container** — Mismatched framework version or CPU/GPU container. Verify your `container_uri`.
   - **Permission denied on GCS** — The training service account can't access your bucket. Check bucket permissions.
3. **Re-test locally** with the same arguments before resubmitting to avoid burning compute time on the same error.

## Training artifacts

After the training run completes, we can manually view our bucket using the Google Cloud Console or run the below code.

```python
total_size_bytes = 0

for blob in client.list_blobs(BUCKET_NAME):
    total_size_bytes += blob.size
    print(blob.name)

total_size_mb = total_size_bytes / (1024**2)
print(f"Total size of bucket '{BUCKET_NAME}': {total_size_mb:.2f} MB")
```

#### Training Artifacts  →  `ARTIFACT_DIR`
This is your *intended output location*, set via `base_output_dir`.  
It contains everything your training script explicitly writes. In our case, this includes:

- **`{BUCKET_NAME}/artifacts/xgb/{RUN_ID}/model/xgboost-model`** — Serialized XGBoost model (Booster) saved via `joblib`; reload later with `joblib.load()` for reuse or deployment.  


#### System-Generated Files
Additional system-generated files (e.g., Vertex's `.tar.gz` code package or `executor_output.json`) will appear under `.vertex_staging/` and can be safely ignored or auto-deleted via lifecycle rules.

## Evaluate the trained model stored on GCS

Now let's verify that the model produced by our Vertex AI job performs identically to the one we trained locally. This time, instead of loading from the local disk, we'll load both the test data and model artifact directly from GCS into memory — the recommended approach for production workflows.

```python
import io

# Load test data directly from GCS into memory
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob("titanic_test.csv")
test_df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))

# Apply same preprocessing logic used during training
X_test, y_test = preprocess_data(test_df)

# Load the model artifact from GCS
MODEL_BLOB_PATH = f"artifacts/xgb/{RUN_ID}/model/xgboost-model"
model_blob = bucket.blob(MODEL_BLOB_PATH)
model_bytes = model_blob.download_as_bytes()
model = joblib.load(io.BytesIO(model_bytes))

# Run predictions and compute accuracy
dtest = xgb.DMatrix(X_test)
y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy (model from Vertex job): {acc:.3f}")
```

::::::::::::::::::::::::::::::::::::::: challenge

### Compare local vs. Vertex AI accuracy

Compare the test accuracy from your local training run with the accuracy from the Vertex AI job. Are they the same? Why or why not?

:::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::: solution

### Solution

The two accuracy values should be very close or identical. Both runs execute the same `train_xgboost.py` script with the same hyperparameters and the same data. XGBoost's `binary:logistic` objective is deterministic given the same input, so the models should produce matching predictions. If they differ, check that you used the same hyperparameter values in both runs and that the data in GCS matches the local copy.

:::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::: challenge

### Explore job logs in the Console

Navigate to **Vertex AI > Training > Custom Jobs** in the Google Cloud Console. Find your most recent job and click on it. Can you locate:

1. The **Logs** tab showing your script's `print()` output?
2. The training time printed by `train_model()`?
3. The output artifact path?

:::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::: solution

### Solution

1. Click your job name, then select the **Logs** tab (or **View logs** link). Your script's `print()` statements — including train/val sizes, training time, and model save path — appear in the log stream.
2. Look for the line `Training time: X.XX seconds` in the logs. This comes from the `train_model()` function in `train_xgboost.py`.
3. The artifact path is shown in the log line `Model saved to gs://...` and also appears in the job details panel under output configuration.

:::::::::::::::::::::::::::::::::::::::

### Looking ahead: when training takes too long

The Titanic dataset is tiny, so our job finishes in minutes. In your real work, you'll encounter datasets and models where a single training run takes hours or days. When that happens, Vertex AI gives you two main levers:

**Option 1: Upgrade to more powerful machine types**
- Use a larger machine or add GPUs (e.g., T4, V100, A100). This is the simplest approach and works well for datasets under ~10 GB.

**Option 2: Use distributed training with multiple replicas**
- Split the dataset across replicas with synchronized gradient updates. This becomes worthwhile when datasets exceed 10–50 GB or single-machine training takes more than 10 hours.

We'll explore both options hands-on in the next episode when we train a PyTorch neural network with GPU acceleration.

::::::::::::::::::::::::::::::::::::: keypoints

- **Environment initialization**: Use `aiplatform.init()` to set defaults for project, region, and bucket.  
- **Local vs managed training**: Test locally before scaling into managed jobs.  
- **Custom jobs**: Vertex AI lets you run scripts as managed training jobs using pre-built or custom containers.  
- **Scaling**: Start small, then scale up to GPUs or distributed jobs as dataset/model size grows.  
- **Monitoring**: Track job logs and artifacts in the Vertex AI Console.  

::::::::::::::::::::::::::::::::::::::::::::::::
