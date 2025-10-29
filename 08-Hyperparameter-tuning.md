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

To conduct efficient hyperparameter tuning with neural networks (or any model) in Vertex AI, we'll use Vertex AI's Hyperparameter Tuning Jobs. The key is defining a clear search space, ensuring metrics are properly logged, and keeping costs manageable by controlling the number of trials and level of parallelization.

### Key steps for hyperparameter tuning

The overall process involves these steps:

1. Prepare the training script and ensure metrics are logged.  
2. Define the hyperparameter search space.  
3. Configure a hyperparameter tuning job in Vertex AI.  
4. Set data paths and launch the tuning job.  
5. Monitor progress in the Vertex AI Console.  
6. Extract the best model and inspect recorded metrics.  

#### 0. Initial setup

Navigate to `/Intro_GCP_for_ML/notebooks/08-Hyperparameter-tuning.ipynb` to begin this notebook. Select the *PyTorch* environment (kernel) Local PyTorch is only needed for local tests. Your *Vertex AI job* uses the container specified by `container_uri` (e.g., `pytorch-cpu.2-1` or `pytorch-gpu.2-1`), so it brings its own framework at run time.

Change to your Jupyter home folder to keep paths consistent.

```python
%cd /home/jupyter/
```

#### 1. Prepare training script with metric logging
Your training script (`train_nn.py`) should periodically print validation metrics in a format Vertex AI can capture. Vertex AI parses lines like `key: value` from stdout.

Add these two lines right after you compute `val_loss` and `val_acc` inside the epoch loop:

```python
print(f"validation_loss: {val_loss:.6f}", flush=True)
print(f"validation_accuracy: {val_acc:.6f}", flush=True)
```

#### 2. Define hyperparameter search space
This step defines which parameters Vertex AI will vary across trials and their allowed ranges. The number of total settings tested is determined later using `max_trial_count`.

Include early-stopping parameters so the tuner can learn good stopping behavior for your dataset:

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

parameter_spec = {
    "learning_rate": hpt.DoubleParameterSpec(min=1e-4, max=1e-1, scale="log"),
    "patience": hpt.IntegerParameterSpec(min=5, max=20, scale="linear"),
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
ARTIFACT_DIR = f"gs://{BUCKET_NAME}/artifacts/pytorch_hpt/{RUN_ID}"

IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-4.py310:latest"  # CPU example
MACHINE = "n1-standard-4"
ACCELERATOR_TYPE = "ACCELERATOR_TYPE_UNSPECIFIED"
ACCELERATOR_COUNT = 0
```

#### 5. Configure hyperparameter tuning job
When you use Vertex AI Hyperparameter Tuning Jobs, each trial needs a complete, runnable training configuration: the script, its arguments, the container image, and the compute environment.  
Rather than defining these pieces inline each time, we create a **CustomJob** to hold that configuration.  

The CustomJob acts as the blueprint for running a single training task — specifying exactly what to run and on what resources. The tuner then reuses that job definition across all trials, automatically substituting in new hyperparameter values for each run.  

This approach has a few practical advantages:

- You only define the environment once — machine type, accelerators, and output directories are all reused across trials.  
- The tuner can safely inject trial-specific parameters (those declared in `parameter_spec`) while leaving other arguments unchanged.  
- It provides a clean separation between *what a single job does* (`CustomJob`) and *how many times to repeat it with new settings* (`HyperparameterTuningJob`).  
- It avoids the extra abstraction layers of higher-level wrappers like `CustomTrainingJob`, which automatically package code and environments. Using `CustomJob.from_local_script` keeps the workflow predictable and explicit.

In short:  
`CustomJob` defines how to run one training run.  
`HyperparameterTuningJob` defines how to repeat it with different parameter sets and track results.  

The number of total runs is set by `max_trial_count`, and the number of simultaneous runs is controlled by `parallel_trial_count`.  Each trial's output and metrics are logged under the GCS `base_output_dir`. **ALWAYS START WITH 2 trials** before scaling up `max_trial_count`.


```python
metric_spec = {"validation_accuracy": "maximize"}  # matches script print key

custom_job = aiplatform.CustomJob.from_local_script(
    display_name=f"{LAST_NAME}_pytorch_hpt-trial_{RUN_ID}",
    script_path="Intro_GCP_for_ML/scripts/train_nn.py",
    container_uri=IMAGE,
    args=[
        f"--train=gs://{BUCKET_NAME}/data/train_data.npz",
        f"--val=gs://{BUCKET_NAME}/data/val_data.npz",
        "--learning_rate=0.001",        # HPT will override when sampling
        "--patience=10",                # HPT will override when sampling
    ],
    base_output_dir=ARTIFACT_DIR,
    machine_type=MACHINE,
    accelerator_type=ACCELERATOR_TYPE,
    accelerator_count=ACCELERATOR_COUNT,
)

DISPLAY_NAME = f"{LAST_NAME}_pytorch_hpt_{RUN_ID}"

# ALWAYS START WITH 2 trials before scaling up `max_trial_count`
tuning_job = aiplatform.HyperparameterTuningJob(
    display_name=DISPLAY_NAME,
    custom_job=custom_job,                 # must be a CustomJob (not CustomTrainingJob)
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=1,                    # controls how many configurations are tested
    parallel_trial_count=2,                # how many run concurrently (keep small for adaptive search)
    # search_algorithm="ALGORITHM_UNSPECIFIED",  # default = adaptive search (Bayesian)
    # search_algorithm="RANDOM_SEARCH",          # optional override
    # search_algorithm="GRID_SEARCH",            # optional override
)

tuning_job.run(sync=True)
print("HPT artifacts base:", ARTIFACT_DIR)
```

#### 6. Monitor tuning job
Open **Vertex AI → Training → Hyperparameter tuning jobs** to track trials, parameters, and metrics. You can also stop jobs from the console if needed. For MLM25, the folllowing link should work: [https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs?hl=en&project=doit-rci-mlm25-4626]([https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs?hl=en&project=doit-rci-mlm25-4626]).

#### 7. Inspect best trial results
After completion, look up the best configuration and objective value from the SDK:

```python
best_trial = tuning_job.trials[0]  # best-first
print("Best hyperparameters:", best_trial.parameters)
print("Best validation_accuracy:", best_trial.final_measurement.metrics)
```

#### 8. Review recorded metrics in GCS
Your script writes a `metrics.json` (with keys such as `final_val_accuracy`, `final_val_loss`) to each trial's output directory (under `ARTIFACT_DIR`). The snippet below aggregates those into a dataframe for side-by-side comparison.

```python
from google.cloud import storage
import json, pandas as pd

def list_metrics_from_gcs(ARTIFACT_DIR: str):
    client = storage.Client()
    bucket_name = ARTIFACT_DIR.replace("gs://", "").split("/")[0]
    prefix = "/".join(ARTIFACT_DIR.replace("gs://", "").split("/")[1:])
    blobs = client.list_blobs(bucket_name, prefix=prefix)

    records = []
    for blob in blobs:
        if blob.name.endswith("metrics.json"):
            trial_id = blob.name.split("/")[-2]
            data = json.loads(blob.download_as_text())
            data["trial_id"] = trial_id
            records.append(data)
    return pd.DataFrame(records)

df = list_metrics_from_gcs(ARTIFACT_DIR)
print(df[["trial_id","final_val_accuracy","final_val_loss","best_val_loss","best_epoch","patience","min_delta","learning_rate"]].sort_values("final_val_accuracy", ascending=False))
```

::::::::::::::::::::::::::::::::::::: discussion

### What is the effect of parallelism in tuning?  

- How might running 10 trials in parallel differ from running 2 at a time in terms of cost, time, and result quality?  
- When would you want to prioritize speed over adaptive search benefits?  

**Cost:**  
- If you run the same total number of trials, total cost is *roughly unchanged*; you're paying for the same amount of compute, just compressed into a shorter wall-clock window.  
- Parallelism can raise short-term spend rate (more machines running at once) and may increase idle/overhead if trials start/finish unevenly.

**Time:**  
- Higher `parallel_trial_count` reduces wall-clock time almost linearly until you hit queue, quota, or data/IO bottlenecks.  
- Startup overhead (image pull, environment setup) is paid for each concurrent trial; with many short trials, this overhead can become a larger fraction of runtime.

**Result quality (adaptive search):**  
- Vertex AI's adaptive search benefits from learning from early trials.  
- With many trials in flight simultaneously, the tuner can't incorporate results quickly, so it explores “blind” for longer. This often yields slightly *worse* final results for a fixed `max_trial_count`.  
- With modest parallelism (e.g., 2–4), the tuner can still update beliefs and exploit promising regions sooner.

**Guidelines:**  
- Start small: `parallel_trial_count` in the range 2–4 is a good default.  
- Keep parallelism to **≤ 25–33%** of `max_trial_count` when you care about adaptive quality.  
- Increase parallelism when your trials are long and you're confident the search space is well-bounded (less need for rapid adaptation).

**When to prioritize speed (higher parallelism):**  
- Strict deadlines or demo timelines.  
- Very cheap/short trials where startup time dominates.  
- You're using a non-adaptive or nearly random search space.  
- You have unused quota/credits and want faster iteration.

**When to prioritize adaptive quality (lower parallelism):**  
- Trials are expensive, noisy, or have high variance; learning from early wins saves budget.  
- Small `max_trial_count` (e.g., ≤ 10–20).  
- Early stopping is enabled and you want the tuner to exploit promising regions quickly.  
- You're adding new dimensions (e.g., LR + patience + min_delta) and want the search to refine intelligently.

**Practical recipe:**  
- First run: `max_trial_count=1`, `parallel_trial_count=1` (pipeline sanity check).  
- Main run: `max_trial_count=10–20`, `parallel_trial_count=2–4`.  
- Scale up parallelism only after the above completes cleanly and you confirm adaptive performance is acceptable.

::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: keypoints

- Vertex AI Hyperparameter Tuning Jobs efficiently explore parameter spaces using adaptive strategies.  
- Define parameter ranges in `parameter_spec`; the number of settings tried is controlled later by `max_trial_count`.  
- Keep the printed metric name consistent with `metric_spec` (here: `validation_accuracy`).  
- Limit `parallel_trial_count` (2–4) to help adaptive search.  
- Use GCS for input/output and aggregate `metrics.json` across trials for detailed analysis.  

::::::::::::::::::::::::::::::::::::::::::::::::
