---
title: "Training Models in Vertex AI: PyTorch Example"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- When should you consider using a GPU or TPU instance for training neural networks in Vertex AI, and what are the benefits and limitations?  
- How does Vertex AI handle distributed training, and which approaches are suitable for typical neural network training?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Preprocess the Titanic dataset for efficient training using PyTorch.  
- Save and upload training and validation data in `.npz` format to GCS.  
- Understand the trade-offs between CPU, GPU, and TPU training for smaller datasets.  
- Deploy a PyTorch model to Vertex AI and evaluate instance types for training performance.  
- Differentiate between data parallelism and model parallelism, and determine when each is appropriate in Vertex AI.  

::::::::::::::::::::::::::::::::::::::::::::::::

## Initial setup

Open a fresh Jupyter notebook in your Vertex AI Workbench environment (e.g., `Training-part2.ipynb`). Then initialize your environment:  

```python
from google.cloud import aiplatform, storage
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

PROJECT_ID = "your-gcp-project-id"
REGION = "us-central1"
BUCKET_NAME = "your-gcs-bucket"

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")
```

- `aiplatform.init()`: Initializes Vertex AI with project, region, and staging bucket.  
- `storage.Client()`: Used to upload training data to GCS.  

## Preparing the data (compressed npz files)

We’ll prepare the Titanic dataset and save as `.npz` files for efficient PyTorch loading.  

```python
# Load and preprocess Titanic dataset
df = pd.read_csv("titanic_train.csv")

df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
y = df['Survived'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

np.savez('train_data.npz', X_train=X_train, y_train=y_train)
np.savez('val_data.npz', X_val=X_val, y_val=y_val)
```

### Upload data to GCS

```python
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

bucket.blob("train_data.npz").upload_from_filename("train_data.npz")
bucket.blob("val_data.npz").upload_from_filename("val_data.npz")

print("Files uploaded to GCS.")
```

:::::::::::::::::::::::::::::::: callout

#### Why use `.npz`?  

- **Optimized data loading**: Compressed binary format reduces I/O overhead.  
- **Batch compatibility**: Works seamlessly with PyTorch `DataLoader`.  
- **Consistency**: Keeps train/validation arrays structured and organized.  
- **Multiple arrays**: Stores multiple arrays (`X_train`, `y_train`) in one file.  

::::::::::::::::::::::::::::::::::::::::::::::::

## Testing locally in notebook

Before scaling up, test your script locally with fewer epochs:  

```python
import torch
import time as t

epochs = 100
learning_rate = 0.001

start_time = t.time()
%run GCP_helpers/train_nn.py --train train_data.npz --val val_data.npz --epochs {epochs} --learning_rate {learning_rate}
print(f"Local training time: {t.time() - start_time:.2f} seconds")
```

## Training via Vertex AI with PyTorch

Vertex AI supports custom training jobs with PyTorch containers.  

```python
job = aiplatform.CustomJob(
    display_name="pytorch-train",
    script_path="GCP_helpers/train_nn.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest",
    requirements=["torch", "pandas", "numpy", "scikit-learn"],
    args=[
        "--train=gs://{}/train_data.npz".format(BUCKET_NAME),
        "--val=gs://{}/val_data.npz".format(BUCKET_NAME),
        "--epochs=1000",
        "--learning_rate=0.001"
    ],
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest",
)

model = job.run(replica_count=1, machine_type="n1-standard-4")
```

## GPU Training in Vertex AI

For small datasets, GPUs may not help. But for larger models/datasets, GPUs (e.g., T4, V100, A100) can reduce training time.  

In your training script (`train_nn.py`), ensure GPU support:  

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Then move models and tensors to `device`.  

## Distributed training in Vertex AI

Vertex AI supports data and model parallelism.  

- **Data parallelism**: Common for neural nets; dataset split across replicas; gradients synced.  
- **Model parallelism**: Splits model across devices, used for very large models.  

```python
model = job.run(replica_count=2, machine_type="n1-standard-8", accelerator_type="NVIDIA_TESLA_T4", accelerator_count=1)
```

## Monitoring jobs

- In the Console: **Vertex AI > Training > Custom Jobs**.  
- Check logs, runtime, and outputs.  
- Cancel jobs as needed.  

::::::::::::::::::::::::::::::::::::: keypoints

- `.npz` files streamline PyTorch data handling and reduce I/O overhead.  
- GPUs may not speed up small models/datasets due to overhead.  
- Vertex AI supports both CPU and GPU training, with scaling via multiple replicas.  
- Data parallelism splits data, model parallelism splits layers — choose based on model size.  
- Test locally first before launching expensive training jobs.  

::::::::::::::::::::::::::::::::::::::::::::::::
