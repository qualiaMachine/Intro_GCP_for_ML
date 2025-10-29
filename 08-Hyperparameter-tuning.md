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

To conduct efficient hyperparameter tuning with neural networks (or any model) in Vertex AI, we’ll use Vertex AI’s **Hyperparameter Tuning Jobs**. The key is defining a clear search space, ensuring metrics are properly logged, and keeping costs manageable by controlling the number of trials and level of parallelization.

### Key steps for hyperparameter tuning

The overall process involves these steps:

1. Prepare training script and ensure metrics are logged.  
2. Define hyperparameter search space.  
3. Configure a hyperparameter tuning job in Vertex AI.  
4. Set data paths and launch the tuning job.  
5. Monitor progress in the Vertex AI Console.  
6. Extract best model and evaluate.  

#### 0. Directory setup
Change directory to your Jupyter home folder.  

```python
%cd /home/jupyter/
```

#### 1. Prepare training script with metric logging
Your training script (`train_nn.py`) should periodically print validation accuracy in a format that Vertex AI can capture.  

```python
if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
    print(f"validation_accuracy: {val_accuracy:.4f}", flush=True)
```

Vertex AI automatically captures metrics logged in this format (`key: value`).  

#### 2. Define hyperparameter search space

In Vertex AI, you specify hyperparameter ranges when configuring the tuning job. You can define both discrete and continuous ranges.

```python
from google.cloud import aiplatform

# Import the right classes directly
from google.cloud.aiplatform import hyperparameter_tuning as hpt

parameter_spec = {
    "epochs": hpt.IntegerParameterSpec(min=100, max=1000, scale="linear"),
    "learning_rate": hpt.DoubleParameterSpec(min=0.001, max=0.1, scale="log"),
}

```

- **IntegerParameterSpec**: Defines integer ranges.  
- **DoubleParameterSpec**: Defines continuous ranges, with optional scaling.  

#### 3. Configure hyperparameter tuning job

```python
from google.cloud import aiplatform, storage
from google.cloud.aiplatform import hyperparameter_tuning as hpt
import datetime as dt

# --- project/region/bucket init ---
client = storage.Client()
PROJECT_ID = client.project
REGION = "us-central1"
BUCKET_NAME = "sinkorswim-johndoe-titanic"  # ADJUST
aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=f"gs://{BUCKET_NAME}/.vertex_staging",
)

# --- run IDs and output dir ---
RUN_ID = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
BASE_DIR = f"gs://{BUCKET_NAME}/artifacts/pytorch_hpt/{RUN_ID}"

# --- training container (use TRAINING image) ---
IMAGE = 'us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-4.py310:latest' # cpu-only version

# --- machine/accelerator for each trial (CPU example) ---
MACHINE = "n1-standard-4"
ACCELERATOR_TYPE = None
ACCELERATOR_COUNT = 0

# --- search space (arg names must match train_nn.py argparse flags) ---
parameter_spec = {
    "epochs": hpt.IntegerParameterSpec(min=100, max=300, scale="linear"),
    "learning_rate": hpt.DoubleParameterSpec(min=1e-4, max=1e-1, scale="log"),
}

metric_spec = {"final_val_loss": "minimize"}  # must match metrics.json key written by your script

# --- build the trial job as a CustomJob; set compute + base_output_dir HERE ---
custom_job = aiplatform.CustomJob.from_local_script(
    display_name=f"pytorch_trial_{RUN_ID}",
    script_path="Intro_GCP_for_ML/scripts/train_nn.py",
    container_uri=IMAGE,
    args=[
        f"--train=gs://{BUCKET_NAME}/data/train_data.npz",
        f"--val=gs://{BUCKET_NAME}/data/val_data.npz",
        "--epochs=200",
        "--learning_rate=0.001",
    ],
    base_output_dir=BASE_DIR,
    machine_type=MACHINE,
    accelerator_type="ACCELERATOR_TYPE_UNSPECIFIED",  # explicitly no GPU
    accelerator_count=0,                              # also required
    requirements=[
        "torch",
        "google-cloud-storage",
        "fsspec",
        "gcsfs",
    ],
)

# --- create and run the HPT job (no base_output_dir or machine args here) ---
tuning_job = aiplatform.HyperparameterTuningJob(
    display_name=f"pytorch_hpt_{RUN_ID}",
    custom_job=custom_job,                 # must be a CustomJob (not CustomTrainingJob)
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=10,
    parallel_trial_count=2,
)

tuning_job.run(sync=True)  # just launch; compute/output were set on the CustomJob above

print("HPT artifacts base:", BASE_DIR)

```

- **max_trial_count**: Total number of configurations tested.  
- **parallel_trial_count**: Number of trials run at once (recommend ≤4 to let adaptive search improve).  

#### 5. Monitor tuning job in Vertex AI Console
1. Navigate to **Vertex AI > Training > Hyperparameter tuning jobs**.  
2. View trial progress, logs, and metrics.  
3. Cancel jobs from the console if needed.  

#### 6. Extract and evaluate the best model

```python
best_trial = hpt_job.trials[0]  # Best trial listed first after completion
print("Best hyperparameters:", best_trial.parameters)
print("Best objective value:", best_trial.final_measurement.metrics)
```

You can then load the best model artifact from the associated GCS path and evaluate on test data.

::::::::::::::::::::::::::::::::::::: discussion

### What is the effect of parallelism in tuning?  

- How might running 10 trials in parallel differ from running 2 at a time in terms of cost, time, and quality of results?  
- When would you want to prioritize speed over adaptive search benefits?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Vertex AI Hyperparameter Tuning Jobs let you efficiently explore parameter spaces using adaptive strategies.  
- Always test with `max_trial_count=1` first to confirm your setup works.  
- Limit `parallel_trial_count` to a small number (2–4) to benefit from adaptive search.  
- Use GCS for input/output and monitor jobs through the Vertex AI Console.  

::::::::::::::::::::::::::::::::::::::::::::::::
