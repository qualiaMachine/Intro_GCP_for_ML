---
title: "Accessing and Managing Data in GCS with Vertex AI Notebooks"
teaching: 20
exercises: 15
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can I load data from GCS into a Vertex AI Workbench notebook?  
- How do I monitor storage usage and costs for my GCS bucket?  
- What steps are involved in pushing new data back to GCS from a notebook?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Read data directly from a GCS bucket into memory in a Vertex AI notebook.  
- Check storage usage and estimate costs for data in a GCS bucket.  
- Upload new files from the Vertex AI environment back to the GCS bucket.  

::::::::::::::::::::::::::::::::::::::::::::::::

## Initial setup

If you haven't already cloned the repository and opened the notebook in the previous episode, do so now: open JupyterLab from your Workbench Instance, run `!git clone https://github.com/qualiaMachine/Intro_GCP_for_ML.git`, and navigate to `/Intro_GCP_for_ML/notebooks/04-Accessing-and-managing-data.ipynb`.

#### Set up GCP environment
Before interacting with GCS, we need to initialize the client libraries. The `storage.Client()` call below creates a connection to Google Cloud Storage using the credentials already attached to your Workbench VM (no manual login needed). Printing the project ID confirms the client initialized correctly and is pointed at the right GCP project.

```python
from google.cloud import storage
client = storage.Client()
print("Project:", client.project)

```


## Reading data from Google Cloud Storage (GCS)

Similar to other cloud vendors, we can either (A) read data directly from Google Cloud Storage (GCS) into memory, or (B) download a copy into your notebook VM. Since we're using notebooks as controllers rather than training environments, the recommended approach is *reading directly from GCS into memory*.

### A) Reading data directly into memory  

```python
import pandas as pd
import io

bucket_name = "sinkorswim-johndoe-titanic" # ADJUST to your bucket's name

bucket = client.bucket(bucket_name)
blob = bucket.blob("titanic_train.csv")
train_data = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
print(train_data.shape)
train_data.head()

```

The Titanic dataset contains passenger information (age, class, fare, etc.) and a binary survival label — we'll train a classifier on this data in Episode 6. Take a moment to explore the dataset:

```python
train_data.info()
train_data.describe()
```

::::::::::::::::::::::::::::::::::::: callout

### Alternative: reading directly with pandas

Vertex AI Workbench comes with `gcsfs` pre-installed, so you can also read directly with `pd.read_csv("gs://your-bucket-name/titanic_train.csv")`. This is convenient for quick exploration. We use the `storage.Client` approach above because it gives you more control (listing blobs, checking sizes, uploading), which you'll need in the sections that follow.

:::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: callout

### Common errors

- **`Forbidden (403)`** — Your service account lacks permission to access the bucket. Revisit the **Adjust bucket permissions** section in [Episode 2](02-Data-storage.md).
- **`NotFound (404)`** — The bucket name or file path is wrong. Double-check `bucket_name` and the blob path with `client.list_blobs(bucket_name)`.
- **`DefaultCredentialsError`** — The notebook cannot find credentials. Make sure you are running on a Vertex AI Workbench Instance (not a local machine).

:::::::::::::::::::::::::::::::::::::

### B) Downloading a local copy

If you prefer, you can download the file to the notebook VM's local disk. This makes repeated reads faster, but may incur small egress costs if the bucket and VM are in different regions (see [Episode 2](02-Data-storage.md) for cost details).

First, check your current working directory so you know where downloaded files will land (you should see `/home/jupyter` or similar):

```python
!pwd
```

```python
blob_name = "titanic_train.csv"
local_path = "/home/jupyter/titanic_train.csv"

bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_name)
blob.download_to_filename(local_path)

!ls -lh /home/jupyter/
```

## Monitoring storage usage and costs

Now that we can read and write data, let's check how much storage our bucket is using and what it costs. It's good practice to periodically check so you can estimate costs before they show up on your bill. The code below iterates over every object in the bucket and sums up their sizes.

```python
total_size_bytes = 0
bucket = client.bucket(bucket_name)

for blob in client.list_blobs(bucket_name):
    total_size_bytes += blob.size

total_size_mb = total_size_bytes / (1024**2)
print(f"Total size of bucket '{bucket_name}': {total_size_mb:.2f} MB")
```

Using the total size, we can estimate monthly costs based on GCS Standard pricing. We include both storage and potential egress (download) costs so you get the full picture.

```python
storage_price_per_gb = 0.02   # $/GB/month for Standard storage
egress_price_per_gb = 0.12    # $/GB for internet egress (same-region transfers are free)
total_size_gb = total_size_bytes / (1024**3)

monthly_storage = total_size_gb * storage_price_per_gb
egress_cost = total_size_gb * egress_price_per_gb  # cost if you download all data once

print(f"Bucket size: {total_size_gb:.4f} GB")
print(f"Estimated monthly storage cost: ${monthly_storage:.4f}")
print(f"Estimated annual storage cost:  ${monthly_storage*12:.4f}")
print(f"One-time full download (egress) cost: ${egress_cost:.4f}")
```

For updated prices, see [GCS Pricing](https://cloud.google.com/storage/pricing).

## Writing output files to GCS
Create a sample file on the notebook VM's storage.

```python
# Create a sample file locally on the notebook VM
file_path = "/home/jupyter/Notes.txt"
with open(file_path, "w") as f:
    f.write("This is a test note for GCS.")

!ls /home/jupyter
```

Upload file.

```python
# Point to the right bucket
bucket = client.bucket(bucket_name)

# Create a *Blob* object, which represents a path inside the bucket
# (here it will end up as gs://<bucket_name>/docs/Notes.txt)
blob = bucket.blob("docs/Notes.txt")

# Upload the local file into that blob (object) in GCS
blob.upload_from_filename(file_path)

print("File uploaded successfully.")

```

List bucket contents:

```python
for blob in client.list_blobs(bucket_name):
    print(blob.name)
```

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Read and explore the test dataset

Read `titanic_test.csv` from your GCS bucket, display its shape, and compare the columns to `train_data`. What column is missing from the test set, and why?

:::::::::::::::: solution

```python
blob = client.bucket(bucket_name).blob("titanic_test.csv")
test_data = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
print("Test shape:", test_data.shape)
print("Train columns:", list(train_data.columns))
print("Test columns:", list(test_data.columns))
test_data.head()
```

The `Survived` column is missing from the test set — that is the label we are trying to predict, so it is intentionally withheld for evaluation.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Upload a summary CSV to GCS

Using `train_data`, compute the survival rate by passenger class (`Pclass`) and upload the result as `results/survival_by_class.csv` to your bucket.

:::::::::::::::: solution

```python
summary = train_data.groupby("Pclass")["Survived"].mean().reset_index()
summary.columns = ["Pclass", "SurvivalRate"]
print(summary)

# Save locally then upload
summary.to_csv("/home/jupyter/survival_by_class.csv", index=False)
blob = client.bucket(bucket_name).blob("results/survival_by_class.csv")
blob.upload_from_filename("/home/jupyter/survival_by_class.csv")
print("Summary uploaded to GCS.")
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 3: Estimating GCS Costs

Suppose you store **50 GB** of data in Standard storage (us-central1) for one month.
- Estimate the monthly storage cost.
- Then estimate the cost if you download (egress) the entire dataset once over the internet at the end of the month.

**Hints**
- Storage: $0.02 per GB-month
- Egress (internet): $0.12 per GB (same-region transfers within GCP are free)

:::::::::::::::: solution

- Storage cost: 50 GB × $0.02 = $1.00
- Egress cost: 50 GB × $0.12 = $6.00
- **Total cost: $7.00 for one month including one full download**

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Load data from GCS into memory with `storage.Client` or directly via `pd.read_csv("gs://...")` to avoid managing local copies.
- Periodically check storage usage and estimate both storage and egress costs to manage your GCS budget.
- Use Vertex AI Workbench notebooks to upload processed results back to GCS for sharing and downstream use.

::::::::::::::::::::::::::::::::::::::::::::::::
