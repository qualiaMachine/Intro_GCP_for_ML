---
title: "Training Models in Vertex AI: PyTorch Example"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions

- When should you consider a GPU (or TPU) instance for PyTorch training in Vertex AI, and what are the trade‑offs for small vs. large workloads?
- How do you launch a script‑based training job and write **all** artifacts (model, metrics, logs) next to each other in GCS without deploying a managed model?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Prepare the Titanic dataset and save train/val arrays to compressed `.npz` files in GCS.
- Submit a *CustomTrainingJob* that runs a PyTorch script and explicitly writes outputs to a chosen `gs://…/artifacts/.../` folder.
- Co‑locate artifacts: `model.pt` (or `.joblib`), `metrics.json`, `eval_history.csv`, and `training.log` for reproducibility.
- Choose CPU vs. GPU instances sensibly; understand when distributed training is (not) worth it.

::::::::::::::::::::::::::::::::::::::::::::::::

## Initial setup

#### 1. Open pre-filled notebook
Navigate to `/Intro_GCP_for_ML/notebooks/06-Training-models-in-VertexAI-GPUs.ipynb` to begin this notebook. Select the *PyTorch* environment (kernel) Local PyTorch is only needed for local tests. Your *Vertex AI job* uses the container specified by `container_uri` (e.g., `pytorch-cpu.2-1` or `pytorch-gpu.2-1`), so it brings its own framework at run time.

#### 2. CD to instance home directory
To ensure we're all in the saming starting spot, change directory to your Jupyter home directory.

```python
%cd /home/jupyter/
```

#### 3. Set environment variables 
This code initializes the Vertex AI environment by importing the Python SDK, setting the project, region, and defining a GCS bucket for input/output data.

```python
from google.cloud import aiplatform, storage
client = storage.Client()
PROJECT_ID = client.project
REGION = "us-central1"
BUCKET_NAME = "sinkorswim-johndoe-titanic" # ADJUST to your bucket's name

print(f"project = {PROJECT_ID}\nregion = {REGION}\nbucket = {BUCKET_NAME}")

# initializes the Vertex AI environment with the correct project and location. Staging bucket is used for storing the compressed software that's packaged for training/tuning jobs.
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}/.vertex_staging") # store tar balls in staging folder 
```

## Prepare data as `.npz`

Why `.npz`? NumPy's `.npz` files are compressed binary containers that can store multiple arrays (e.g., features and labels) together in a single file. They offer numerous benefits:

- Smaller, faster I/O than CSV for arrays.  
- One file can hold multiple arrays (`X_train`, `y_train`).
- Natural fit for `torch.utils.data.Dataset` / `DataLoader`.  
- **Cloud-friendly:** compressed `.npz` files reduce upload and download times and minimize GCS egress costs. Because each `.npz` is a single binary object, reading it from Google Cloud Storage (GCS) requires only one network call—much faster and cheaper than streaming many small CSVs or images individually.  
- **Efficient data movement:** when you launch a Vertex AI training job, GCS objects referenced in your script (for example, `gs://.../train_data.npz`) are automatically staged to the job's VM or container at runtime. Vertex copies these objects into its local scratch disk before execution, so subsequent reads (e.g., `np.load(...)`) occur from local storage rather than directly over the network. For small-to-medium datasets, this happens transparently and incurs minimal startup delay.  
- **Reproducible binary format:** unlike CSV, `.npz` preserves exact dtypes and shapes, ensuring identical results across different environments and containers.  
- Each GCS object read or listing request incurs a small per-request cost; using a single `.npz` reduces both the number of API calls and associated latency.

  
```python
import pandas as pd
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Titanic CSV (from local or GCS you've already downloaded to the notebook)
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob("titanic_train.csv")
df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))

# Minimal preprocessing to numeric arrays
sex_enc = LabelEncoder().fit(df["Sex"])            # Fit label encoder on 'Sex' column (male/female)
df["Sex"] = sex_enc.transform(df["Sex"])           # Convert 'Sex' to numeric values (e.g., male=1, female=0)
df["Embarked"] = df["Embarked"].fillna("S")       # Replace missing embarkation ports with most common ('S')
emb_enc = LabelEncoder().fit(df["Embarked"])       # Fit label encoder on 'Embarked' column (S/C/Q)
df["Embarked"] = emb_enc.transform(df["Embarked"]) # Convert embarkation categories to numeric codes
df["Age"] = df["Age"].fillna(df["Age"].median())   # Fill missing ages with median (robust to outliers)
df["Fare"] = df["Fare"].fillna(df["Fare"].median())# Fill missing fares with median to avoid NaNs

X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]].values  # Select numeric feature columns as input
y = df["Survived"].values                                                # Target variable (1=survived, 0=did not survive)

scaler = StandardScaler()                                                # Initialize standard scaler for standardization (best practice for neural net training)
X = scaler.fit_transform(X)                                              # Scale features to mean=0, std=1 for stable training

X_train, X_val, y_train, y_val = train_test_split(                       # Split dataset into training and validation sets
    X, y, test_size=0.2, random_state=42)                                # 80% training, 20% validation (fixed random seed)

np.savez("/home/jupyter/train_data.npz", X_train=X_train, y_train=y_train)             # Save training arrays to compressed .npz file
np.savez("/home/jupyter/val_data.npz",   X_val=X_val,   y_val=y_val)                   # Save validation arrays to compressed .npz file

```

We can then upload the files to our GCS bucket.

```python
# Upload to GCS
bucket.blob("data/train_data.npz").upload_from_filename("/home/jupyter/train_data.npz")
bucket.blob("data/val_data.npz").upload_from_filename("/home/jupyter/val_data.npz")
print("Uploaded: gs://%s/data/train_data.npz and val_data.npz" % BUCKET_NAME)
```

To check our work (bucket contents), we can again use the following code:

```python
total_size_bytes = 0
# bucket = client.bucket(BUCKET_NAME)

for blob in client.list_blobs(BUCKET_NAME):
    total_size_bytes += blob.size
    print(blob.name)

total_size_mb = total_size_bytes / (1024**2)
print(f"Total size of bucket '{BUCKET_NAME}': {total_size_mb:.2f} MB")
```

## Minimal PyTorch training script (`train_nn.py`) - local test

**Outside of this workshop, you should run these kinds of tests on your local laptop or lab PC when possible.** We're using the Workbench VM here only for convenience in this workshop setting, but this does incur a small fee for our running VM. 

- For large datasets, use a small representative sample of the total dataset when testing locally (i.e., just to verify that code is working and model overfits nearly perfectly after training enough epochs)
- For larger models, use smaller model equivalents (e.g., 100M vs 7B params) when testing locally
  
Find this file in our repo: `Intro_GCP_for_ML/scripts/train_nn.py`. It does three things:
1) loads `.npz` from local or GCS
2) trains a tiny multilayer perceptron (MLP)
3) writes all outputs side‑by‑side (model + metrics + eval history + training.log) to the same `--model_out` folder.

To test this code, we can run the following:

```python
import time as t

start = t.time()

# Example: run your custom training script with args
!python /home/jupyter/Intro_GCP_for_ML/scripts/train_nn.py \
    --train /home/jupyter/train_data.npz \
    --val /home/jupyter/val_data.npz \
    --epochs 50 \
    --learning_rate 0.001

print(f"Total local runtime: {t.time() - start:.2f} seconds")
```

If applicable (numpy mismatch), run the below code after uncommenting it (select code and type `Ctrl+/` for multiline uncommenting)

```python
# # Fix numpy mismatch
# !pip install --upgrade --force-reinstall "numpy<2"

# # Then, rerun:

# import time as t

# start = t.time()

# # Example: run your custom training script with args
# !python /home/jupyter/Intro_GCP_for_ML/scripts/train_nn.py \
#     --train /home/jupyter/train_data.npz \
#     --val /home/jupyter/val_data.npz \
#     --epochs 50 \
#     --learning_rate 0.001

# print(f"Total local runtime: {t.time() - start:.2f} seconds")
```

### Reproducibility test
Without reproducibility, it's impossible to gain reliable insights into the efficacy of our methods. An essential component of applied ML/AI is ensuring our experiments are reproducible. Let's first rerun the same code we did above to verify we get the same result. 

* Take a look near the top of `Intro_GCP_for_ML/scripts/train_nn.py` where we are setting multiple numpy and torch seeds to ensure reproducibility.

```python
import time as t

start = t.time()

# Example: run your custom training script with args
!python /home/jupyter/Intro_GCP_for_ML/scripts/train_nn.py \
    --train /home/jupyter/train_data.npz \
    --val /home/jupyter/val_data.npz \
    --epochs 50 \
    --learning_rate 0.001

print(f"Total local runtime: {t.time() - start:.2f} seconds")
```

**Please don't use cloud resources for code that is not reproducible!**

## Launch the training job 

In the previous episode, we trained an XGBoost model using Vertex AI's CustomTrainingJob interface. Here, we'll do the same for a PyTorch neural network. The structure is nearly identical —  we define a training script, select a prebuilt container (CPU or GPU), and specify where to write all outputs in Google Cloud Storage (GCS). The main difference is that PyTorch requires us to save our own model weights and metrics inside the script rather than relying on Vertex to package a model automatically.

### Set training job configuration vars
For our image, we can find the corresponding PyTorch image by visiting: [cloud.google.com/vertex-ai/docs/training/pre-built-containers#pytorch](https://cloud.google.com/vertex-ai/docs/training/pre-built-containers#pytorch)

```python
RUN_ID = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
ARTIFACT_DIR = f"gs://{BUCKET_NAME}/artifacts/pytorch/{RUN_ID}"
IMAGE = 'us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-4.py310:latest' # cpu-only version
MACHINE = "n1-standard-4" # CPU fine for small datasets

print(f"RUN_ID = {RUN_ID}\nARTIFACT_DIR = {ARTIFACT_DIR}\nMACHINE = {MACHINE}")
```

### Init the training job with configurations

```python
# init job (this does not consume any resources)
LAST_NAME = 'DOE' # REPLACE with your last name. Since we're in a shared account envirnoment, this will help us track down jobs in the Console
DISPLAY_NAME = f"{LAST_NAME}_pytorch_nn_{RUN_ID}" 

# init the job. This does not consume resources until we run job.run()
job = aiplatform.CustomTrainingJob(
    display_name=DISPLAY_NAME,
    script_path="Intro_GCP_for_ML/scripts/train_nn.py",
    container_uri=IMAGE)

```

### Run the job, paying for our `MACHINE` on-demand.

```python
job.run(
    args=[
        f"--train=gs://{BUCKET_NAME}/data/train_data.npz",
        f"--val=gs://{BUCKET_NAME}/data/val_data.npz",
        "--epochs=200",
        "--learning_rate=0.001"
    ],
    replica_count=1,
    machine_type=MACHINE,
    base_output_dir=ARTIFACT_DIR,  # sets AIP_MODEL_DIR used by your script
    sync=True,
)
```

**What you'll see in `gs://…/artifacts/pytorch/<RUN_ID>/`:**
- `model.pt` — PyTorch weights (`state_dict`).
- `metrics.json` — final val loss, hyperparameters, dataset sizes, device, model URI.
- `eval_history.csv` — per‑epoch validation loss (for plots/regression checks).
- `training.log` — complete stdout/stderr for reproducibility and debugging.

## Optional: GPU training

For larger models or heavier data:

```python
job = aiplatform.CustomTrainingJob(
    display_name=f"pytorch_nn_gpu_{RUN_ID}",
    script_path="GCP_helpers/train_nn.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest",
    requirements=["torch", "numpy", "fsspec", "gcsfs"],
)

job.run(
    args=[
        f"--train=gs://{BUCKET_NAME}/data/train_data.npz",
        f"--val=gs://{BUCKET_NAME}/data/val_data.npz",
        f"--epochs=200",
        f"--learning_rate=0.001",
        f"--model_out={MODEL_URI}",
    ],
    replica_count=1,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    sync=True,
)
```

GPU tips:
- On small problems, GPU startup/transfer overhead can erase speedups—benchmark before you scale.
- Stick to a single replica unless your batch sizes and dataset really warrant data parallelism.

## Distributed training (when to consider)

- **Data parallelism** (DDP) helps when a single GPU is saturated by batch size/throughput. For most workshop‑scale models, a single machine/GPU is simpler and cheaper.
- **Model parallelism** is for very large networks that don't fit on one device—overkill for this lesson.

## Monitoring jobs & finding outputs

- Console → Vertex AI → Training → Custom Jobs → your run → “Output directory” shows the container logs and the environment's `AIP_MODEL_DIR`.
- Your script writes **model + metrics + eval history + training.log** next to `--model_out`, e.g., `gs://<bucket>/artifacts/pytorch/<RUN_ID>/`.

## Additional resources
To learn more about PyTorch and Vertex AI integrations, visit the docs: [docs.cloud.google.com/vertex-ai/docs/start/pytorch](https://docs.cloud.google.com/vertex-ai/docs/start/pytorch)

::::::::::::::::::::::::::::::::::::: keypoints

- Use **CustomTrainingJob** with a prebuilt PyTorch container; let your script control outputs via `--model_out`.
- Keep artifacts **together** (model, metrics, history, log) in one folder for reproducibility.
- `.npz` speeds up loading and plays nicely with PyTorch.
- Start on CPU for small datasets; use GPU only when profiling shows a clear win.
- Skip `base_output_dir` unless you specifically want Vertex's default run directory; staging bucket is just for the SDK packaging tarball.

::::::::::::::::::::::::::::::::::::::::::::::::

