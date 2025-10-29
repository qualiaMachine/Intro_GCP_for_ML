---
title: "Hyperparameter Tuning in Vertex AI: Neural Network Example"
teaching: 60
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can we efficiently manage hyperparameter tuning in Vertex AI?  
- How can we parallelize tuning jobs to optimize time without increasing costs?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Set up and run a hyperparameter tuning job in Vertex AI.  
- Define search spaces for `ContinuousParameter` and `CategoricalParameter`.  
- Log and capture objective metrics for evaluating tuning success.  
- Optimize tuning setup to balance cost and efficiency, including parallelization.  

::::::::::::::::::::::::::::::::::::::::::::::::

To conduct efficient hyperparameter tuning with neural networks (or any model) in Vertex AI, we’ll use Vertex AI’s Hyperparameter Tuning Jobs. The key is defining a clear search space, ensuring metrics are properly logged, and keeping costs manageable by controlling the number of trials and level of parallelization.

### Key steps for hyperparameter tuning

The overall process involves these steps:

1. Prepare the training script and ensure metrics are logged.  
2. Define the hyperparameter search space.  
3. Configure a hyperparameter tuning job in Vertex AI.  
4. Set data paths and launch the tuning job.  
5. Monitor progress in the Vertex AI Console.  
6. Extract the best model and inspect recorded metrics.  

#### 0. Directory setup
Change to your Jupyter home folder to keep paths consistent.

```python
%cd /home/jupyter/
```

#### 1. Prepare training script with metric logging
Your training script (`train_nn.py`) should periodically print validation metrics in a format Vertex AI can capture. Vertex AI parses lines like `key: value` from stdout.

Add these two lines right after you compute `val_loss` and `val_acc` inside the epoch loop (the patch below shows exactly where):

```python
print(f"validation_loss: {val_loss:.6f}", flush=True)
print(f"validation_accuracy: {val_acc:.6f}", flush=True)
```

This is in addition to your existing human-readable line (e.g., `epoch=... val_loss:... val_acc:...`).  
Patch file you can apply: `train_nn.patch` (provided below). The current script also writes a `metrics.json` with keys like `final_val_accuracy` which we will read later. fileciteturn0file0

#### 2. Define hyperparameter search space
This step defines which parameters Vertex AI will vary across trials and their allowed ranges. The number of total settings tested is determined later using `max_trial_count`.

Include early-stopping parameters so the tuner can learn good stopping behavior for your dataset:

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

parameter_spec = {
    "epochs": hpt.IntegerParameterSpec(min=100, max=300, scale="linear"),
    "learning_rate": hpt.DoubleParameterSpec(min=1e-4, max=1e-1, scale="log"),
    "patience": hpt.IntegerParameterSpec(min=5, max=20, scale="linear"),
    "min_delta": hpt.DoubleParameterSpec(min=0.0, max=0.01, scale="linear"),  # improvement threshold on val_loss
    # You can also add restore_best as categorical if you want to compare behaviors:
    # "restore_best": hpt.CategoricalParameterSpec(values=["true","false"]),
}
```

#### 3. Initialize Vertex AI, project, and bucket
Initialize the Vertex AI SDK and set your staging and artifact locations in GCS.

```python
from google.cloud import aiplatform, storage
import datetime as dt

client = storage.Client()
PROJECT_ID = client.project
REGION = "us-central1"
LAST_NAME = "DOE"  # change to your name or unique ID
BUCKET_NAME = "sinkorswim-johndoe-titanic"  # replace with your bucket name

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=f"gs://{BUCKET_NAME}/.vertex_staging",
)
```

#### 4. Define runtime configuration
Create a unique run ID and set the container, machine type, and base output directory for artifacts.

```python
RUN_ID = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
BASE_DIR = f"gs://{BUCKET_NAME}/artifacts/pytorch_hpt/{RUN_ID}"

IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-4.py310:latest"  # CPU example
MACHINE = "n1-standard-4"
ACCELERATOR_TYPE = "ACCELERATOR_TYPE_UNSPECIFIED"
ACCELERATOR_COUNT = 0
```

#### 5. Configure hyperparameter tuning job
Set the optimization metric to the printed key `validation_accuracy`. Start with one trial to validate your setup before scaling.

```python
metric_spec = {"validation_accuracy": "maximize"}  # matches script print key

custom_job = aiplatform.CustomJob.from_local_script(
    display_name=f"{LAST_NAME}_pytorch_hpt-trial_{RUN_ID}",
    script_path="Intro_GCP_for_ML/scripts/train_nn.py",
    container_uri=IMAGE,
    args=[
        f"--train=gs://{BUCKET_NAME}/data/train_data.npz",
        f"--val=gs://{BUCKET_NAME}/data/val_data.npz",
        "--epochs=200",                 # HPT will override when sampling
        "--learning_rate=0.001",        # HPT will override when sampling
        "--patience=10",                # HPT will override when sampling
        "--min_delta=0.0",              # HPT will override when sampling
        # "--restore_best=true",        # optional categorical param if enabled above
    ],
    base_output_dir=BASE_DIR,
    machine_type=MACHINE,
    accelerator_type=ACCELERATOR_TYPE,
    accelerator_count=ACCELERATOR_COUNT,
)

DISPLAY_NAME = f"{LAST_NAME}_pytorch_hpt_{RUN_ID}"

tuning_job = aiplatform.HyperparameterTuningJob(
    display_name=DISPLAY_NAME,
    custom_job=custom_job,                 # must be a CustomJob (not CustomTrainingJob)
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=10,                    # controls how many configurations are tested
    parallel_trial_count=2,                # how many run concurrently (keep small for adaptive search)
)

tuning_job.run(sync=True)
print("HPT artifacts base:", BASE_DIR)
```

#### 6. Monitor tuning job
Open **Vertex AI → Training → Hyperparameter tuning jobs** to track trials, parameters, and metrics. You can also stop jobs from the console if needed.

#### 7. Inspect best trial results
After completion, look up the best configuration and objective value from the SDK:

```python
best_trial = tuning_job.trials[0]  # best-first
print("Best hyperparameters:", best_trial.parameters)
print("Best validation_accuracy:", best_trial.final_measurement.metrics)
```

#### 8. Review recorded metrics in GCS
Your script writes a `metrics.json` (with keys such as `final_val_accuracy`, `final_val_loss`) to each trial’s output directory (under `BASE_DIR`). The snippet below aggregates those into a dataframe for side-by-side comparison.

```python
from google.cloud import storage
import json, pandas as pd

def list_metrics_from_gcs(base_dir: str):
    client = storage.Client()
    bucket_name = base_dir.replace("gs://", "").split("/")[0]
    prefix = "/".join(base_dir.replace("gs://", "").split("/")[1:])
    blobs = client.list_blobs(bucket_name, prefix=prefix)

    records = []
    for blob in blobs:
        if blob.name.endswith("metrics.json"):
            trial_id = blob.name.split("/")[-2]
            data = json.loads(blob.download_as_text())
            data["trial_id"] = trial_id
            records.append(data)
    return pd.DataFrame(records)

df = list_metrics_from_gcs(BASE_DIR)
print(df[["trial_id","final_val_accuracy","final_val_loss","best_val_loss","best_epoch","patience","min_delta","learning_rate"]].sort_values("final_val_accuracy", ascending=False))
```

::::::::::::::::::::::::::::::::::::: discussion

### What is the effect of parallelism in tuning?  

- How might running 10 trials in parallel differ from running 2 at a time in terms of cost, time, and result quality?  
- When would you want to prioritize speed over adaptive search benefits?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Vertex AI Hyperparameter Tuning Jobs efficiently explore parameter spaces using adaptive strategies.  
- Define parameter ranges in `parameter_spec`; the number of settings tried is controlled later by `max_trial_count`.  
- Keep the printed metric name consistent with `metric_spec` (here: `validation_accuracy`).  
- Limit `parallel_trial_count` (2–4) to help adaptive search.  
- Use GCS for input/output and aggregate `metrics.json` across trials for detailed analysis.  

::::::::::::::::::::::::::::::::::::::::::::::::
