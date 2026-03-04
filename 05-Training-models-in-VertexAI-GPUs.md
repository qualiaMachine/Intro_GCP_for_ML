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
Navigate to `/Intro_GCP_for_ML/notebooks/05-Training-models-in-VertexAI-GPUs.ipynb` to begin this notebook. Select the *PyTorch* environment (kernel) Local PyTorch is only needed for local tests. Your *Vertex AI job* uses the container specified by `container_uri` (e.g., `pytorch-cpu.2-5` or `pytorch-gpu.2-5`), so it brings its own framework at run time.

#### 2. CD to instance home directory
To ensure we're all in the same starting spot, change directory to your Jupyter home directory.

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
BUCKET_NAME = "doe-titanic" # ADJUST to your bucket's name
LAST_NAME = 'DOE' # ADJUST to your last name. Since we're in a shared account environment, this will help us track down jobs in the Console

print(f"project = {PROJECT_ID}\nregion = {REGION}\nbucket = {BUCKET_NAME}")

# initializes the Vertex AI environment with the correct project and location. Staging bucket is used for storing the compressed software that's packaged for training/tuning jobs.
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}/.vertex_staging") # store tar balls in staging folder 
```

## Prepare data as `.npz`

Unlike the XGBoost script from Episode 4 (which handles preprocessing internally from raw CSV), our PyTorch script expects pre-processed NumPy arrays. We'll prepare those here and save them as `.npz` files.

Why `.npz`? NumPy's `.npz` files are compressed binary containers that can store multiple arrays (e.g., features and labels) together in a single file:

- **Compact & fast:** smaller than CSV, and one file can hold multiple arrays (`X_train`, `y_train`).
- **Cloud-friendly:** each `.npz` is a single GCS object — one network call to read instead of streaming many small files, reducing latency and egress costs.
- **Vertex AI integration:** when you launch a training job, GCS objects are automatically staged to the job VM's local scratch disk, so `np.load(...)` reads from local storage at runtime.
- **Reproducible:** unlike CSV, `.npz` preserves exact dtypes and shapes across environments.

  
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
sex_enc = LabelEncoder().fit(df["Sex"])
df["Sex"] = sex_enc.transform(df["Sex"])
df["Embarked"] = df["Embarked"].fillna("S")
emb_enc = LabelEncoder().fit(df["Embarked"])
df["Embarked"] = emb_enc.transform(df["Embarked"])
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]].values
y = df["Survived"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

np.savez("/home/jupyter/train_data.npz", X_train=X_train, y_train=y_train)
np.savez("/home/jupyter/val_data.npz",   X_val=X_val,   y_val=y_val)

```

We can then upload the files to our GCS bucket.

```python
# Upload to GCS
bucket.blob("data/train_data.npz").upload_from_filename("/home/jupyter/train_data.npz")
bucket.blob("data/val_data.npz").upload_from_filename("/home/jupyter/val_data.npz")
print("Uploaded: gs://%s/data/train_data.npz and val_data.npz" % BUCKET_NAME)
```

Verify the upload by listing your bucket contents (same pattern as Episode 3):

```python
for blob in client.list_blobs(BUCKET_NAME):
    print(blob.name)
```

## Minimal PyTorch training script (`train_nn.py`) - local test

**Outside of this workshop, you should run these kinds of tests on your local laptop or lab PC when possible.** We're using the Workbench VM here only for convenience in this workshop setting, but this does incur a small fee for our running VM. 

- For large datasets, use a small representative sample of the total dataset when testing locally (i.e., just to verify that code is working and model overfits nearly perfectly after training enough epochs)
- For larger models, use smaller model equivalents (e.g., 100M vs 7B params) when testing locally
  
Find this file in our repo: `Intro_GCP_for_ML/scripts/train_nn.py`. It does three things:
1) loads `.npz` from local or GCS paths (transparently handles both)
2) trains a small neural network (a 3-layer MLP) with early stopping
3) writes all outputs side‑by‑side (model + metrics + eval history + training.log) to the folder specified by the `AIP_MODEL_DIR` environment variable (set automatically by Vertex AI via `base_output_dir`), falling back to the current directory for local runs.

::::::::::::::::::::::::::::::::::::: callout
### What's inside `train_nn.py`? (Quick reference)
You don't need to understand every line of the PyTorch code for this workshop — the focus is on how to package and run *any* training script on Vertex AI. That said, here's a quick orientation:

- **GCS helpers** (top of file): `read_npz_any()` and `save_*_any()` functions detect `gs://` paths and use the GCS Python client automatically. This is the key pattern that makes the same script work locally and in the cloud.
- **`AIP_MODEL_DIR`**: Vertex AI sets this environment variable to tell your script where to write artifacts. The script reads it at the top of `main()`.
- **Model**: A small feedforward network (`TitanicNet`) — the architecture details aren't important for this lesson.
- **Early stopping**: Training halts when validation loss stops improving (controlled by `--patience`). This saves compute time and cost on cloud jobs.
:::::::::::::::::::::::::::::::::::::::::::::::::

To test this code, we can run the following:

```python
# configure training hyperparameters to use in all model training runs downstream
MAX_EPOCHS = 500
LR =  0.001
PATIENCE = 50

# local training run
import time as t

start = t.time()

# Example: run your custom training script with args
!python /home/jupyter/Intro_GCP_for_ML/scripts/train_nn.py \
    --train /home/jupyter/train_data.npz \
    --val /home/jupyter/val_data.npz \
    --epochs $MAX_EPOCHS \
    --learning_rate $LR \
    --patience $PATIENCE

print(f"Total local runtime: {t.time() - start:.2f} seconds")
```

::::::::::::::::::::::::::::::::::::::: callout
### NumPy version mismatch?
If the cell above fails with a NumPy error (e.g., `module 'numpy' has no attribute ...`), run this fix and then re-run the training cell:

```python
!pip install --upgrade --force-reinstall "numpy<2"
```
The PyTorch kernel occasionally ships with NumPy 2.x, which has breaking API changes.
::::::::::::::::::::::::::::::::::::::

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
    --epochs $MAX_EPOCHS \
    --learning_rate $LR \
    --patience $PATIENCE

print(f"Total local runtime: {t.time() - start:.2f} seconds")
```

**Please don't use cloud resources for code that is not reproducible!**

### Evaluate the locally trained model on the validation data

Let's load the model we just trained and run it against the validation set. This confirms the saved weights produce the expected accuracy before we move to cloud training.

```python
import sys, torch, numpy as np
sys.path.append("/home/jupyter/Intro_GCP_for_ML/scripts")
from train_nn import TitanicNet

# load validation data
d = np.load("/home/jupyter/val_data.npz")
X_val, y_val = d["X_val"], d["y_val"]

# tensors
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

# rebuild model and load weights
m = TitanicNet()
state = torch.load("/home/jupyter/model.pt", map_location="cpu", weights_only=True)
m.load_state_dict(state)
m.eval()

with torch.no_grad():
    probs = m(X_val_t).squeeze(1)                # [N], sigmoid outputs in (0,1)
    preds_t = (probs >= 0.5).long()              # [N] int64
    correct = (preds_t == y_val_t).sum().item()
    acc = correct / y_val_t.shape[0]

print(f"Local model val accuracy: {acc:.4f}")

```

We should see an accuracy that matches our best epoch in the local training run. Note that in our setup, early stopping is based on validation loss; not accuracy.

## Launch the training job 

In the previous episode, we trained an XGBoost model using Vertex AI's CustomTrainingJob interface. Here, we'll do the same for a PyTorch neural network. The structure is nearly identical —  we define a training script, select a prebuilt container (CPU or GPU), and specify where to write all outputs in Google Cloud Storage (GCS). The main difference is that PyTorch requires us to save our own model weights and metrics inside the script rather than relying on Vertex to package a model automatically.

### Set training job configuration vars

::::::::::::::::::::::::::::::::::::: callout
### Check supported container versions
Google periodically retires older prebuilt images. Before running the cells below, verify that the PyTorch version in `IMAGE` is still listed at [Prebuilt containers for training](https://cloud.google.com/vertex-ai/docs/training/pre-built-containers#pytorch). If you see an `INVALID_ARGUMENT` error about an unsupported image, update the version number (e.g., `2-5` → `2-6`).
:::::::::::::::::::::::::::::::::::::

```python
import datetime as dt
RUN_ID = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
ARTIFACT_DIR = f"gs://{BUCKET_NAME}/artifacts/pytorch/{RUN_ID}"
IMAGE = 'us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.2-5.py310:latest' # cpu-only version
MACHINE = "n1-standard-4" # CPU fine for small datasets

print(f"RUN_ID = {RUN_ID}\nARTIFACT_DIR = {ARTIFACT_DIR}\nMACHINE = {MACHINE}")
```

### Init the training job with configurations

```python
# init job (this does not consume any resources)
DISPLAY_NAME = f"{LAST_NAME}_pytorch_nn_{RUN_ID}" 
print(DISPLAY_NAME)

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
        f"--epochs={MAX_EPOCHS}",
        f"--learning_rate={LR}",
        f"--patience={PATIENCE}",
    ],
    replica_count=1,
    machine_type=MACHINE,
    base_output_dir=ARTIFACT_DIR,  # sets AIP_MODEL_DIR used by your script
    sync=True,
)
print("Artifacts folder:", ARTIFACT_DIR)
```
#### Monitoring training jobs in the Console
1. Go to the Google Cloud Console.  
2. Navigate to **Vertex AI > Training > Custom Jobs**.  
3. Click on your job name to see status, logs, and output model artifacts.  
4. Cancel jobs from the console if needed (be careful not to stop jobs you don't own in shared projects).

**Quick link** (replace `YOUR_PROJECT_ID`): `https://console.cloud.google.com/vertex-ai/training/training-pipelines?project=YOUR_PROJECT_ID`

After the job completes, your training script writes several output files to the GCS artifact directory. Here's what you'll find in `gs://…/artifacts/pytorch/<RUN_ID>/`:

- `model.pt` — PyTorch weights (`state_dict`).
- `metrics.json` — final val loss, hyperparameters, dataset sizes, device, model URI.
- `eval_history.csv` — per‑epoch validation loss (for plots/regression checks).
- `training.log` — complete stdout/stderr for reproducibility and debugging.

### Evaluate the Vertex-trained model on the validation data

We can check our work to see if this model gives the same result as our "locally" trained model above.

To follow best practices, we will simply load this model into memory from GCS.

```python
import sys, torch, numpy as np
sys.path.append("/home/jupyter/Intro_GCP_for_ML/scripts")
from train_nn import TitanicNet

# -----------------
# download model.pt straight into memory and load weights
# -----------------

ARTIFACT_PREFIX = f"artifacts/pytorch/{RUN_ID}/model"

MODEL_PATH = f"{ARTIFACT_PREFIX}/model.pt"
model_blob = bucket.blob(MODEL_PATH)
model_bytes = model_blob.download_as_bytes()

# load from bytes
model_pt = io.BytesIO(model_bytes)

# rebuild model and load weights
state = torch.load(model_pt, map_location="cpu", weights_only=True)
m = TitanicNet()
m.load_state_dict(state)
m.eval();
```

Evaluate using the same pattern from the CPU evaluation section above — load validation data from GCS, run predictions, and check accuracy. The results should match the CPU job since we set random seeds.

```python
# Read validation data from GCS (reuses val data from local eval above)
VAL_PATH = "data/val_data.npz"
val_blob = bucket.blob(VAL_PATH)
val_bytes = val_blob.download_as_bytes()
d = np.load(io.BytesIO(val_bytes))
X_val, y_val = d["X_val"], d["y_val"]
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

with torch.no_grad():
    probs = m(X_val_t).squeeze(1)
    preds_t = (probs >= 0.5).long()
    correct = (preds_t == y_val_t).sum().item()
    acc = correct / y_val_t.shape[0]

print(f"Vertex model val accuracy: {acc:.4f}")
```

## GPU-Accelerated Training on Vertex AI

Our CPU job above worked fine for this small dataset. In practice, you'd switch to a GPU when training takes too long on CPU — typically with larger models (millions of parameters) or larger datasets (hundreds of thousands of rows). For the Titanic dataset, the GPU will likely be *slower* end-to-end due to provisioning overhead, but we'll run it here to learn the workflow.

The changes from CPU to GPU are minimal — this is one of the advantages of Vertex AI's container-based approach:

- The container image switches to the GPU-enabled version (`pytorch-gpu.2-5.py310:latest`), which includes CUDA and cuDNN.
- The machine type (`n1-standard-8`) defines CPU and memory resources, while we add a GPU accelerator (`NVIDIA_TESLA_T4`, `NVIDIA_L4`, etc.). **For guidance on selecting a machine type and accelerator, visit the [Compute for ML](https://qualiamachine.github.io/Intro_GCP_for_ML/compute-for-ML.html) resource.**
- The training script, arguments, and artifact handling all stay the same.

::::::::::::::::::::::::::::::::::::: callout
### GPU quota unavailable?
If your job fails with a quota error, don't worry — re-run using the CPU configuration from the previous section. The results will be identical, just slower. GPU quota requests can take 1–3 business days to process.
:::::::::::::::::::::::::::::::::::::

```python
from google.cloud import aiplatform

RUN_ID = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

# GCS folder where ALL artifacts (model.pt, metrics.json, eval_history.csv, training.log) will be saved.
# Your train_nn.py writes to AIP_MODEL_DIR, and base_output_dir (below) sets that variable for the job.
ARTIFACT_DIR = f"gs://{BUCKET_NAME}/artifacts/pytorch/{RUN_ID}"

# ---- Container image ----
# Use a prebuilt TRAINING image that has PyTorch + CUDA. This enables GPU at runtime.
IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-5.py310:latest"

# ---- Machine vs Accelerator (important!) ----
# machine_type = the VM's CPU/RAM shape. It is NOT a GPU by itself.
# We often pick n1-standard-8 as a balanced baseline for single-GPU jobs.
MACHINE = "n1-standard-8"

# To actually get a GPU, you *attach* one via accelerator_type + accelerator_count.
# Common choices:
#   "NVIDIA_TESLA_T4" (cost-effective, widely available)
#   "NVIDIA_L4"       (newer, CUDA 12.x, good perf/$)
#   "NVIDIA_TESLA_V100" / "NVIDIA_A100_40GB" (high-end, pricey)
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1  # Increase (2,4) only if your code supports multi-GPU (e.g., DDP)

# Alternative (GPU-bundled) machines:
# If you pick an A2 type like "a2-highgpu-1g", it already includes 1 A100 GPU.
# In that case, you can omit accelerator_type/accelerator_count entirely.
# Example:
# MACHINE = "a2-highgpu-1g"
# (and then remove the accelerator_* kwargs in job.run)

print(
    "RUN_ID =", RUN_ID,
    "\nARTIFACT_DIR =", ARTIFACT_DIR,
    "\nIMAGE =", IMAGE,
    "\nMACHINE =", MACHINE,
    "\nACCELERATOR_TYPE =", ACCELERATOR_TYPE,
    "\nACCELERATOR_COUNT =", ACCELERATOR_COUNT,
)

DISPLAY_NAME = f"{LAST_NAME}_pytorch_nn_{RUN_ID}"

job = aiplatform.CustomTrainingJob(
    display_name=DISPLAY_NAME,
    script_path="Intro_GCP_for_ML/scripts/train_nn.py",  # Your PyTorch trainer
    container_uri=IMAGE,  # Must be a *training* image (not prediction)
)

job.run(
    args=[
        f"--train=gs://{BUCKET_NAME}/data/train_data.npz",
        f"--val=gs://{BUCKET_NAME}/data/val_data.npz",
        f"--epochs={MAX_EPOCHS}",
        f"--learning_rate={LR}",
        f"--patience={PATIENCE}",
    ],
    replica_count=1,                 # One worker (simple, cheaper)
    machine_type=MACHINE,            # CPU/RAM shape of the VM (no GPU implied)
    accelerator_type=ACCELERATOR_TYPE,   # Attaches the selected GPU model
    accelerator_count=ACCELERATOR_COUNT, # Number of GPUs to attach
    base_output_dir=ARTIFACT_DIR,    # Sets AIP_MODEL_DIR used by your script for all artifacts
    sync=True,                       # Waits for job to finish so you can inspect outputs immediately
)

print("Artifacts folder:", ARTIFACT_DIR)
```

Just as we did for the CPU job, let's evaluate the GPU-trained model to confirm it produces the same accuracy. We load the model weights directly from GCS into memory.

```python
import sys, torch, numpy as np
sys.path.append("/home/jupyter/Intro_GCP_for_ML/scripts")
from train_nn import TitanicNet

# -----------------
# download model.pt straight into memory and load weights
# -----------------

ARTIFACT_PREFIX = f"artifacts/pytorch/{RUN_ID}/model"

MODEL_PATH = f"{ARTIFACT_PREFIX}/model.pt"
model_blob = bucket.blob(MODEL_PATH)
model_bytes = model_blob.download_as_bytes()

# load from bytes
model_pt = io.BytesIO(model_bytes)

# rebuild model and load weights
state = torch.load(model_pt, map_location="cpu", weights_only=True)
m = TitanicNet()
m.load_state_dict(state)
m.eval();
```

Evaluate the GPU model using the same pattern — results should match because we set random seeds in `train_nn.py`.

```python
with torch.no_grad():
    probs = m(X_val_t).squeeze(1)
    preds_t = (probs >= 0.5).long()
    correct = (preds_t == y_val_t).sum().item()
    acc = correct / y_val_t.shape[0]

print(f"GPU model val accuracy: {acc:.4f}")
```

:::::::::::::::::::::::::::::::::::::::: challenge

### Cloud workflow review

Now that you've run both a CPU and GPU training job, answer the following:

1. **Artifact location**: Where did Vertex AI write your model artifacts? How does `base_output_dir` in `job.run()` relate to the `AIP_MODEL_DIR` environment variable inside the container?
2. **CPU vs. GPU job time**: Compare the wall-clock times of your CPU and GPU jobs (visible in the Console under **Vertex AI > Training > Custom Jobs**). Which was faster? Why might the GPU job be *slower* for this dataset?
3. **Container choice**: We used `pytorch-cpu.2-5.py310` for the CPU job and `pytorch-gpu.2-5.py310` for the GPU job. What would happen if you used the CPU container but still passed `accelerator_type` and `accelerator_count`?
4. **Cost awareness**: You used `n1-standard-4` for CPU and `n1-standard-8` + T4 for GPU. Using the [Compute for ML](https://qualiamachine.github.io/Intro_GCP_for_ML/compute-for-ML.html) resource, estimate the relative hourly cost difference between these configurations.

:::::::::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::::: solution

### Solution

1. `base_output_dir` tells the Vertex AI SDK to set the `AIP_MODEL_DIR` environment variable inside the training container. Your script reads `os.environ.get("AIP_MODEL_DIR", ".")` and writes all artifacts there. The result is everything lands under `gs://<bucket>/artifacts/pytorch/<RUN_ID>/model/`.
2. For the small Titanic dataset (~700 training rows), the CPU job is typically faster end-to-end. GPU jobs incur extra overhead: provisioning the accelerator, loading CUDA libraries, and transferring data to the GPU. GPU acceleration pays off when training itself is the bottleneck (larger models, larger batches).
3. The job would either fail or ignore the GPU. The CPU container doesn't include CUDA/cuDNN, so even if a GPU is attached to the VM, PyTorch can't use it. Always match your container image to your hardware configuration.
4. Approximate on-demand rates (us-central1): `n1-standard-4` is ~ `$0.19`/hr; `n1-standard-8` + 1x T4 is ~ `$0.54`/hr (VM) + ~ `$0.35`/hr (T4) = ~ `$0.89`/hr total. The GPU configuration is roughly 4–5x more expensive per hour — worth it only when training speedup exceeds that cost ratio.

:::::::::::::::::::::::::::::::::::::::

### GPU and scaling considerations

- On small problems, GPU startup/transfer overhead can erase speedups — benchmark before you scale.
- Stick to a single GPU unless your workload genuinely saturates it. Multi-GPU (data parallelism / DDP) and model parallelism exist for large-scale training but add significant complexity and cost — well beyond this workshop's scope.

## Additional resources
To learn more about PyTorch and Vertex AI integrations, visit the docs: [docs.cloud.google.com/vertex-ai/docs/start/pytorch](https://docs.cloud.google.com/vertex-ai/docs/start/pytorch)

::::::::::::::::::::::::::::::::::::: keypoints

- Use **CustomTrainingJob** with a prebuilt PyTorch container; your script reads `AIP_MODEL_DIR` (set automatically by `base_output_dir`) to know where to write artifacts.
- Keep artifacts **together** (model, metrics, history, log) in one GCS folder for reproducibility.
- `.npz` is a compact, cloud-friendly format — one GCS read per file, preserves exact dtypes.
- Start on CPU for small datasets; add a GPU only when training time justifies the extra provisioning overhead and cost.
- `staging_bucket` is just for the SDK's packaging tarball — `base_output_dir` is where your script's actual artifacts go.

::::::::::::::::::::::::::::::::::::::::::::::::

