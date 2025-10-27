---
title: "Accessing and Managing Data in GCS with Vertex AI Notebooks"
teaching: 20
exercises: 10
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

#### Load pre-filled Jupyter notebooks 
If you haven't opened your newly created VM from the last episode yet, lick **Open JupyterLab** to open the instance in Jupyter Lab. From there, we can create as many Jupyter notebooks as we would like within the instance environment. 

We will then select the standard python3 environment to start our first .ipynb notebook (Jupyter notebook). We can use this environment since we aren't doing any training/tuning just yet.

Within the Jupyter notebook, run the following command to clone the lesson repo into our Jupyter environment:

```sh
!git clone https://github.com/qualiaMachine/Intro_GCP_for_ML.git
```

Then, navigate to `/Intro_GCP_for_ML/notebooks/Accessing-and-managing-data.ipynb` to begin the first notebook.

#### Set up GCP environment
Before interacting with GCS, we need to authenticate and initialize the client libraries. This ensures our notebook can talk to GCP securely.

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

bucket_name = "sinkorswim-johndoe-titanic" # ADJUST to your bucket's name

bucket = client.bucket(bucket_name)
blob = bucket.blob("titanic_train.csv")
train_data = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
print(train_data.shape)
train_data.head()

```

If you get an error, return to the Google Cloud Console (where we created our bucket and VM) and search for "Cloud Shell Editor". Open a shell editor and run the below command, *replacing the bucket name with your bucket's name`:

```sh
gcloud storage buckets add-iam-policy-binding gs://sinkorswim-johndoe-titanic \
  --member="serviceAccount:549047673858-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

### B) Downloading a local copy  

If you prefer, you can download the file from your bucket to the notebook VM's local disk. This makes repeated reads faster within our notebook environment, but note that *each download counts as a "GET" request* and may incur a small data transfer (egress) cost *if the bucket and VM are in different regions*. If both are in the same region, there are no transfer fees — only standard request costs (typically fractions of a cent).

Let's verify what our path looks like first.

```
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

## Checking storage usage of a bucket

```python
total_size_bytes = 0
bucket = client.bucket(bucket_name)

for blob in client.list_blobs(bucket_name):
    total_size_bytes += blob.size

total_size_mb = total_size_bytes / (1024**2)
print(f"Total size of bucket '{bucket_name}': {total_size_mb:.2f} MB")
```

## Estimating storage costs

```python
storage_price_per_gb = 0.02  # $/GB/month for Standard storage
total_size_gb = total_size_bytes / (1024**3)
monthly_cost = total_size_gb * storage_price_per_gb

print(f"Estimated monthly cost: ${monthly_cost:.4f}")
print(f"Estimated annual cost: ${monthly_cost*12:.4f}")
```

For updated prices, see [GCS Pricing](https://cloud.google.com/storage/pricing).

## Writing output files to GCS

```python
# Create a sample file locally on the notebook VM
with open("Notes.txt", "w") as f:
    f.write("This is a test note for GCS.")

# Point to the right bucket
bucket = client.bucket(bucket_name)

# Create a *Blob* object, which represents a path inside the bucket
# (here it will end up as gs://<bucket_name>/docs/Notes.txt)
blob = bucket.blob("docs/Notes.txt")

# Upload the local file into that blob (object) in GCS
blob.upload_from_filename("Notes.txt")

print("File uploaded successfully.")

```

List bucket contents:

```python
for blob in client.list_blobs(bucket_name):
    print(blob.name)
```

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge: Estimating GCS Costs

Suppose you store **50 GB** of data in Standard storage (us-central1) for one month.  
- Estimate the monthly storage cost.  
- Then estimate the cost if you download (egress) the entire dataset once at the end of the month.  

**Hints**  
- Storage: $0.02 per GB-month  
- Egress: $0.12 per GB  

:::::::::::::::: solution

- Storage cost: 50 GB × $0.02 = $1.00  
- Egress cost: 50 GB × $0.12 = $6.00  
- **Total cost: $7.00 for one month including one full download**  

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints 

- Load data from GCS into memory to avoid managing local copies when possible.  
- Periodically check storage usage and costs to manage your GCS budget.  
- Use Vertex AI Workbench notebooks to upload analysis results back to GCS, keeping workflows organized and reproducible.  

::::::::::::::::::::::::::::::::::::::::::::::::
