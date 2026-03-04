---
title: "Hyperparameter Tuning in Vertex AI: Neural Network Example"
teaching: 40
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can we efficiently manage hyperparameter tuning in Vertex AI?  
- How can we parallelize tuning jobs to optimize time without increasing costs?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Set up and run a hyperparameter tuning job in Vertex AI.  
- Define search spaces using `DoubleParameterSpec` and `IntegerParameterSpec`.  
- Log and capture objective metrics for evaluating tuning success.  
- Optimize tuning setup to balance cost and efficiency, including parallelization.  

::::::::::::::::::::::::::::::::::::::::::::::::

In the previous episode (Episode 5) you submitted a single PyTorch training job to Vertex AI and inspected its artifacts. That gave you one model trained with one set of hyperparameters. In practice, choices like learning rate, early-stopping patience, and regularization thresholds can dramatically affect model quality — and the best combination is rarely obvious up front.

In this episode we'll use Vertex AI's **Hyperparameter Tuning Jobs** to systematically search for better settings. The key is defining a clear search space, ensuring metrics are properly logged, and keeping costs manageable by controlling the number of trials and level of parallelization.

### Key steps for hyperparameter tuning

The overall process involves these steps:

1. Prepare the training script and ensure metrics are logged.  
2. Define the hyperparameter search space.  
3. Configure a hyperparameter tuning job in Vertex AI.  
4. Set data paths and launch the tuning job.  
5. Monitor progress in the Vertex AI Console.  
6. Extract the best model and inspect recorded metrics.  

## Initial setup

#### 1. Open pre-filled notebook
Navigate to `/Intro_GCP_for_ML/notebooks/06-Hyperparameter-tuning.ipynb` to begin this notebook. **Select the *PyTorch* environment (kernel).** Local PyTorch is only needed for local tests — your *Vertex AI job* uses the container specified by `container_uri` (e.g., `pytorch-gpu.2-6`), so it brings its own framework at run time.

#### 2. CD to instance home directory
Change to your Jupyter home folder to keep paths consistent.

```python
%cd /home/jupyter/
```

## Prepare and configure the tuning job

#### 3. Understand how the training script reports metrics
Your training script (`train_nn.py`) **already includes** hyperparameter tuning metric reporting — you don't need to modify it. Here's how it works:

The script uses the `cloudml-hypertune` library (pre-installed on Vertex AI training workers) to report metrics so the tuner can compare trials. A `try/except` block lets the same script run locally without crashing:

```python
# Already in train_nn.py — initialization near the top:
try:
    from hypertune import HyperTune
    _hpt = HyperTune()
    _hpt_enabled = True
except Exception:
    _hpt = None
    _hpt_enabled = False
```

Inside the training loop, after computing validation metrics each epoch:

```python
# Already in train_nn.py — inside the epoch loop:
if _hpt_enabled:
    _hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="validation_accuracy",
        metric_value=val_acc,
        global_step=ep,
    )
```

The critical detail: the `hyperparameter_metric_tag` string **must exactly match** the key you use in `metric_spec` when configuring the tuning job (e.g., `"validation_accuracy"`). If they don't match, trials will show as **INFEASIBLE**.

#### 4. Define hyperparameter search space
This step defines which parameters Vertex AI will vary across trials and their allowed ranges. The number of total settings tested is determined later using `max_trial_count`.

Vertex AI uses **Bayesian optimization** by default (internally listed as `"ALGORITHM_UNSPECIFIED"` in the API).  That means if you don’t explicitly specify a search algorithm, Vertex AI automatically applies an adaptive Bayesian strategy to balance exploration (trying new areas of the parameter space) and exploitation (focusing near the best results so far).  Each completed trial helps the tuner model how your objective metric (for example, `validation_accuracy`) changes across parameter values. Subsequent trials then sample new parameter combinations that are statistically more likely to improve performance, which usually yields better results than random or grid search—especially when `max_trial_count` is limited.

Vertex AI supports four parameter spec types. This episode uses the first two:

| Spec type | Use case | Example |
|---|---|---|
| `DoubleParameterSpec` | Continuous floats | Learning rate 1e-4 to 1e-2 |
| `IntegerParameterSpec` | Whole numbers | Patience 5 to 20 |
| `DiscreteParameterSpec` | Specific numeric values | Batch size [32, 64, 128] |
| `CategoricalParameterSpec` | Named options (strings) | Optimizer ["adam", "sgd"] |

Include early-stopping parameters so the tuner can learn good stopping behavior for your dataset:

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

parameter_spec = {
    "learning_rate": hpt.DoubleParameterSpec(min=1e-4, max=1e-2, scale="log"),
    "patience": hpt.IntegerParameterSpec(min=5, max=20, scale="linear"),
    "min_delta": hpt.DoubleParameterSpec(min=1e-6, max=1e-3, scale="log"),
}
```

#### 5. Initialize Vertex AI, project, and bucket
Initialize the Vertex AI SDK and set your staging and artifact locations in GCS.

```python
from google.cloud import aiplatform, storage
import datetime as dt

client = storage.Client()
PROJECT_ID = client.project
REGION = "us-central1"
LAST_NAME = "DOE"  # change to your name or unique ID
BUCKET_NAME = "doe-titanic"  # replace with your bucket name

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=f"gs://{BUCKET_NAME}/.vertex_staging",
)
```

#### 6. Define runtime configuration
Create a unique run ID and set the container, machine type, and base output directory for artifacts. Each variable controls a different aspect of the training environment:

- **`RUN_ID`** — a timestamp that uniquely identifies this tuning session, used to organize artifacts in GCS.
- **`ARTIFACT_DIR`** — the GCS folder where all trial outputs (models, metrics, logs) will be written.
- **`IMAGE`** — the prebuilt Docker container that includes PyTorch and its dependencies.
- **`MACHINE`** — the VM shape (CPU/RAM) for each trial. Start small for testing.
- **`ACCELERATOR_TYPE` / `ACCELERATOR_COUNT`** — set to unspecified/0 for CPU-only runs. Change these to attach a GPU when needed.

```python
RUN_ID = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
ARTIFACT_DIR = f"gs://{BUCKET_NAME}/artifacts/pytorch_hpt/{RUN_ID}"

IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-6.py310:latest"  # XLA container includes cloudml-hypertune
MACHINE = "n1-standard-4"
ACCELERATOR_TYPE = "ACCELERATOR_TYPE_UNSPECIFIED"
ACCELERATOR_COUNT = 0
```

#### 7. Configure hyperparameter tuning job
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

The number of total runs is set by `max_trial_count`, and the number of simultaneous runs is controlled by `parallel_trial_count`.  Each trial's output and metrics are logged under the GCS `base_output_dir`. **ALWAYS START WITH 1 trial** before scaling up `max_trial_count`.


```python
# metric_spec = {"validation_loss": "minimize"} - also stored by train_nn.py
metric_spec = {"validation_accuracy": "maximize"}

custom_job = aiplatform.CustomJob.from_local_script(
    display_name=f"{LAST_NAME}_pytorch_hpt-trial_{RUN_ID}",
    script_path="Intro_GCP_for_ML/scripts/train_nn.py",
    container_uri=IMAGE,
    requirements=["python-json-logger>=2.0.7"],  # resolves a dependency conflict in the prebuilt container
    args=[
        f"--train=gs://{BUCKET_NAME}/data/train_data.npz",
        f"--val=gs://{BUCKET_NAME}/data/val_data.npz",
        "--learning_rate=0.001",        # HPT will override when sampling
        "--patience=10",                # HPT will override when sampling
        "--min_delta=0.001",            # HPT will override when sampling
    ],
    base_output_dir=ARTIFACT_DIR,
    machine_type=MACHINE,
    accelerator_type=ACCELERATOR_TYPE,
    accelerator_count=ACCELERATOR_COUNT,
)

DISPLAY_NAME = f"{LAST_NAME}_pytorch_hpt_{RUN_ID}"

# ALWAYS START WITH 1 trial before scaling up `max_trial_count`
tuning_job = aiplatform.HyperparameterTuningJob(
    display_name=DISPLAY_NAME,
    custom_job=custom_job,                 # must be a CustomJob (not CustomTrainingJob)
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=1,                    # controls how many configurations are tested
    parallel_trial_count=1,                # how many run concurrently (keep small for adaptive search)
    # search_algorithm="ALGORITHM_UNSPECIFIED",  # default = adaptive search (Bayesian)
    # search_algorithm="RANDOM_SEARCH",          # optional override
    # search_algorithm="GRID_SEARCH",            # optional override
)

tuning_job.run(sync=True)
print("HPT artifacts base:", ARTIFACT_DIR)
```

## Run and analyze results

#### 8. Monitor tuning job
Open **Vertex AI → Training → Hyperparameter tuning jobs** in the Cloud Console to track trials, parameters, and metrics. You can also stop jobs from the console if needed.

> **Note:** Replace the project ID in the URL below with your own if you are not using the shared workshop project.

For the MLM25 workshop: [Hyperparameter tuning jobs](https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs?hl=en&project=doit-rci-mlm25-4626).

::::::::::::::::::::::::::::::::::::::: callout

### Troubleshooting common HPT issues

- **All trials show INFEASIBLE:** The `hyperparameter_metric_tag` in your training script doesn't match the key in `metric_spec`. Double-check spelling and case — `"validation_accuracy"` is not `"val_accuracy"`.
- **Quota errors on launch:** Your project may not have enough VM or GPU quota in the selected region. Check **IAM & Admin → Quotas** and request an increase or switch to a smaller `MACHINE` type.
- **Trial succeeds but metrics are empty:** Make sure `cloudml-hypertune` is importable inside the container. The prebuilt PyTorch containers include it. If using a custom container, add `cloudml-hypertune` to your `requirements`.
- **Job stuck in PENDING:** Another tuning or training job may be consuming your quota. Check **Vertex AI → Training** for running jobs.

:::::::::::::::::::::::::::::::::::::::::::::::

#### 9. Inspect best trial results
After completion, look up the best configuration and objective value from the SDK:

```python
best_trial = tuning_job.trials[0]  # best-first
print("Best hyperparameters:", best_trial.parameters)
print("Best validation_accuracy:", best_trial.final_measurement.metrics)
```

#### 10. Review recorded metrics in GCS
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
cols = ["trial_id","final_val_accuracy","final_val_loss","best_val_loss",
        "best_epoch","patience","min_delta","learning_rate"]
df_sorted = df[cols].sort_values("final_val_accuracy", ascending=False)
print(df_sorted)
```

#### 11. Visualize trial comparison
A quick chart makes it easier to see which trials performed best and how learning rate relates to accuracy:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart: accuracy per trial
axes[0].barh(df_sorted["trial_id"].astype(str), df_sorted["final_val_accuracy"])
axes[0].set_xlabel("Validation Accuracy")
axes[0].set_ylabel("Trial")
axes[0].set_title("Accuracy by Trial")

# Scatter: learning rate vs accuracy (color = patience)
sc = axes[1].scatter(
    df_sorted["learning_rate"], df_sorted["final_val_accuracy"],
    c=df_sorted["patience"], cmap="viridis", edgecolors="k", s=80,
)
axes[1].set_xscale("log")
axes[1].set_xlabel("Learning Rate (log scale)")
axes[1].set_ylabel("Validation Accuracy")
axes[1].set_title("LR vs. Accuracy (color = patience)")
plt.colorbar(sc, ax=axes[1], label="patience")

plt.tight_layout()
plt.show()
```

::::::::::::::::::::::::::::::::::::: challenge

### Exercise 1: Widen the learning-rate search space

The current search space uses `min=1e-4, max=1e-2` for learning rate. Suppose you suspect that slightly larger learning rates (up to `0.1`) might converge faster with early stopping enabled.

1. Update `parameter_spec` to widen the `learning_rate` range to `max=0.1`.
2. Thinking question: Why does `scale="log"` make sense for learning rate but `scale="linear"` makes sense for patience?
3. **Do not run the job yet** — just update the configuration.

::::::::::::::::::::::: solution

```python
parameter_spec = {
    "learning_rate": hpt.DoubleParameterSpec(min=1e-4, max=1e-1, scale="log"),
    "patience": hpt.IntegerParameterSpec(min=5, max=20, scale="linear"),
    "min_delta": hpt.DoubleParameterSpec(min=1e-6, max=1e-3, scale="log"),
}
```

**Why log vs. linear?** Learning rate values span several orders of magnitude (0.0001 to 0.1), so `scale="log"` ensures the tuner samples evenly across those orders rather than clustering near the high end. Patience is an integer (5–20) where each step is equally meaningful, so `scale="linear"` is appropriate.

:::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Exercise 2: Scale up trials

After verifying your single-trial sanity check completed successfully, modify the tuning job configuration to run a proper search:

1. Set `max_trial_count=6` and `parallel_trial_count=2`.
2. Before running, estimate the approximate cost: if each trial takes ~5 minutes on an `n1-standard-4` (~ `$0.19`/hr), how much would 6 trials cost?
3. Run the updated job and monitor it in the Vertex AI Console.

::::::::::::::::::::::: solution

```python
tuning_job = aiplatform.HyperparameterTuningJob(
    display_name=DISPLAY_NAME,
    custom_job=custom_job,
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=6,
    parallel_trial_count=2,
)
```

**Cost estimate:** 6 trials x 5 min each = 30 minutes of compute. At ~ `$0.19`/hr for `n1-standard-4`, that's roughly `$0.10` total. With `parallel_trial_count=2`, wall-clock time would be approximately 15 minutes (3 batches of 2 trials). The adaptive search can still learn between batches since parallelism is kept low relative to total trials.

:::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: discussion

### What is the effect of parallelism in tuning?

- How might running 10 trials in parallel differ from running 2 at a time in terms of cost, time, and result quality?
- When would you want to prioritize speed over adaptive search benefits?

| Factor | High parallelism (e.g., 10) | Low parallelism (e.g., 2) |
|---|---|---|
| **Wall-clock time** | Shorter | Longer |
| **Total cost** | ~Same (slightly more overhead) | ~Same |
| **Adaptive search quality** | Worse (tuner explores “blind”) | Better (tuner learns between batches) |
| **Best for** | Cheap/short trials, deadlines | Expensive trials, small budgets |

**Why does parallelism hurt result quality?** Vertex AI's adaptive search learns from completed trials to choose better parameter combinations. With many trials in flight simultaneously, the tuner can't incorporate results quickly — it explores “blind” for longer, often yielding slightly worse results for a fixed `max_trial_count`. With modest parallelism (2–4), the tuner can update beliefs and exploit promising regions between batches.

**Guidelines:**
- Keep `parallel_trial_count` to **≤ 25–33%** of `max_trial_count` when you care about adaptive quality.
- Increase parallelism when trials are long and the search space is well-bounded.

::::::::::::::::::::::::::::::::::::::: callout

### When to prioritize speed vs. adaptive quality

**Favor higher parallelism** when you have strict deadlines, very cheap/short trials where startup time dominates, a non-adaptive search, or unused quota/credits.

**Favor lower parallelism** when trials are expensive or noisy, `max_trial_count` is small (≤ 10–20), early stopping is enabled, or you're exploring many dimensions at once.

:::::::::::::::::::::::::::::::::::::::::::::::

**Practical recipe:**
- First run: `max_trial_count=1`, `parallel_trial_count=1` (pipeline sanity check).
- Main run: `max_trial_count=10–20`, `parallel_trial_count=2–4`.
- Scale up parallelism only after the above completes cleanly and you confirm adaptive performance is acceptable.

::::::::::::::::::::::::::::::::::::::::::::::::


## What's next: using your tuned model

After tuning, your best model's weights sit in GCS under the best trial's artifact directory. The most common next steps are:

- **Batch prediction (most common):** Load the best model from GCS and run inference on a dataset — this is what we did in the evaluation sections of Episodes 4–5 when we loaded models from GCS into memory. For larger-scale batch prediction, Vertex AI offers [Batch Prediction Jobs](https://cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions) that handle provisioning and scaling automatically.
- **Experiment tracking:** Vertex AI [Experiments](https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments) can log metrics, parameters, and artifacts across runs for systematic comparison. Consider integrating this into your workflow as your projects grow.
- **Online deployment:** If you need real-time predictions via an API, Vertex AI [Endpoints](https://cloud.google.com/vertex-ai/docs/predictions/get-online-predictions) let you deploy your model — but endpoints bill continuously (~ `$4.50`/day for an `n1-standard-4`), so only deploy when you genuinely need a live API.


::::::::::::::::::::::::::::::::::::: keypoints

- Vertex AI Hyperparameter Tuning Jobs efficiently explore parameter spaces using adaptive strategies.
- Define parameter ranges in `parameter_spec`; the number of settings tried is controlled later by `max_trial_count`.
- The `hyperparameter_metric_tag` reported by `cloudml-hypertune` must exactly match the key in `metric_spec`.
- Limit `parallel_trial_count` (2–4) to help adaptive search.
- Use GCS for input/output and aggregate `metrics.json` across trials for detailed analysis.

::::::::::::::::::::::::::::::::::::::::::::::::
