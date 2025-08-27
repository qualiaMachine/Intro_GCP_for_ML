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
parameter_spec = {
    "epochs": aiplatform.hyperparameter_tuning_utils.IntegerParameterSpec(min=100, max=1000, scale="linear"),
    "learning_rate": aiplatform.hyperparameter_tuning_utils.DoubleParameterSpec(min=0.001, max=0.1, scale="log")
}
```

- **IntegerParameterSpec**: Defines integer ranges.  
- **DoubleParameterSpec**: Defines continuous ranges, with optional scaling.  

#### 3. Configure hyperparameter tuning job

```python
from google.cloud import aiplatform

job = aiplatform.CustomJob(
    display_name="pytorch-train-hpt",
    script_path="GCP_helpers/train_nn.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest",
    requirements=["torch", "pandas", "numpy", "scikit-learn"],
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest",
)

hpt_job = aiplatform.HyperparameterTuningJob(
    display_name="pytorch-hpt-job",
    custom_job=job,
    metric_spec={"validation_accuracy": "maximize"},
    parameter_spec=parameter_spec,
    max_trial_count=4,
    parallel_trial_count=2,
)
```

#### 4. Launch the hyperparameter tuning job

```python
hpt_job.run(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=[
        "--train=gs://{}/train_data.npz".format(BUCKET_NAME),
        "--val=gs://{}/val_data.npz".format(BUCKET_NAME),
        "--epochs=100",
        "--learning_rate=0.001"
    ]
)
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
