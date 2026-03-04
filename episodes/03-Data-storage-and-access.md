---
title: "Data Storage and Access"
teaching: 35
exercises: 15
---

:::::::::::::::::::::::::::::::::::::: questions

- How can I store and manage data effectively in GCP for Vertex AI workflows?
- What are the advantages of Google Cloud Storage (GCS) compared to local or VM storage for machine learning projects?
- How can I load data from GCS into a Vertex AI Workbench notebook?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain data storage options in GCP for machine learning projects.
- Set up a GCS bucket and upload data.
- Read data directly from a GCS bucket into memory in a Vertex AI notebook.
- Monitor storage usage and estimate costs.
- Upload new files from the Vertex AI environment back to the GCS bucket.

::::::::::::::::::::::::::::::::::::::::::::::::

ML/AI projects rely on data, making efficient storage and management essential. Google Cloud offers several storage options, but the most common for ML/AI workflows are **Virtual Machine (VM) disks** and **Google Cloud Storage (GCS) buckets**.

> #### Consult your institution's IT before handling sensitive data in GCP
> As with AWS, **do not upload restricted or sensitive data to GCP services unless explicitly approved by your institution's IT or cloud security team**. For regulated datasets (HIPAA, FERPA, proprietary), work with your institution to ensure encryption, restricted access, and compliance with policies.

## Options for storage: VM Disks or GCS

### What is a VM  disk?
A VM disk is the storage volume attached to a Compute Engine VM or a Vertex AI Workbench notebook. It can store datasets and intermediate results, but it is tied to the lifecycle of the VM.

#### When to store data directly on a VM disk

- Useful for small, temporary datasets processed interactively.
- Data persists if the VM is stopped, but storage costs continue as long as the disk exists.
- Not ideal for collaboration, scaling, or long-term dataset storage.

::::::::::::::::::::::::::::::::::::: callout

#### Limitations of VM disk storage

- **Scalability**: Limited by disk size quota.
- **Sharing**: Harder to share across projects or team members.
- **Cost**: More expensive per GB compared to GCS for long-term storage.

::::::::::::::::::::::::::::::::::::::::::::::::

### What is a GCS bucket?
For most ML/AI workflows in GCP, **Google Cloud Storage (GCS) buckets** are recommended. A GCS bucket is a container in Google's object storage service where you can store an essentially unlimited number of files. Data in GCS can be accessed from Vertex AI training jobs, Workbench notebooks, and other GCP services using a *GCS URI* (e.g., `gs://your-bucket-name/your-file.csv`).

::::::::::::::::::::::::::::::::::::: callout

#### GCS URIs — your cloud file paths
GCS URIs follow the format `gs://bucket-name/path/to/file.csv`. Think of them as cloud file paths. You'll use these URIs throughout the workshop to reference data in training scripts, notebooks, and SDK calls.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: callout

#### Benefits of using GCS (recommended for ML/AI workflows)

- **Separation of storage and compute**: Data remains available even if VMs or notebooks are deleted.
- **Easy sharing**: Buckets can be accessed by collaborators with the right IAM roles.
- **Integration with Vertex AI and BigQuery**: Read and write data directly using other GCP tools.
- **Scalability**: Handles datasets of any size without disk limits.
- **Cost efficiency**: Lower cost than persistent disks (VM storage) for long-term storage.
- **Data persistence**: Durable and highly available across regions.
- **Filesystem mounting**: GCS buckets can be mounted as local directories using [Cloud Storage FUSE](https://cloud.google.com/storage/docs/cloud-storage-fuse/overview), making them accessible like regular filesystems for tools that expect local file paths.

::::::::::::::::::::::::::::::::::::::::::::::::

## Creating a GCS bucket

### 1. Sign in to Google Cloud Console

- Go to [console.cloud.google.com](https://console.cloud.google.com/) and log in with your credentials. <!-- shared workshop project URL: https://console.cloud.google.com/welcome?project=doit-rci-mlm25-4626 -->
- Select your project from the project dropdown at the top of the page. If you're using the shared workshop project, the instructor will provide the project name.

### 2. Navigate to Cloud Storage

- In the search bar, type **Storage**.
- Click **Cloud Storage > Buckets**.

### 3. Create a new bucket

- Click **Create bucket** and configure the following settings:

- **Bucket name**: Enter a globally unique name using the convention `firstlastname-dataname` (e.g., `johndoe-titanic`).
- **Labels**: Add cost-tracking labels — `name=firstname-lastname`, `purpose=bucket-dataname`. In shared accounts this is *mandatory*.
- **Location**: Choose **Region** → `us-central1` (same region as your compute to avoid egress charges).
- **Storage class**: **Standard** (best for active ML/AI workflows).
- **Access control**: **Uniform** (simpler IAM-based permissions).
- **Protection**: Leave default **soft delete** enabled; skip versioning and retention policies.

![Final GCS Bucket Settings](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/bucket-fullyconfigured.jpg){alt="Recommended GCS bucket settings."}

Click **Create** if everything looks good.

### 4. Upload files to the bucket

- If you haven't yet, download the data for this workshop (Right-click → Save as):
   [data.zip](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/data/data.zip)
  - Extract the zip folder contents (Right-click → Extract all on Windows; double-click on macOS).
  - The zip contains the **Titanic dataset** — passenger information (age, class, fare, etc.) with a survival label. This is a classic binary classification task we'll use for training in later episodes.
- In the bucket dashboard, click **Upload Files**.
- Select your Titanic CSVs (`titanic_train.csv` and `titanic_test.csv`) and upload.

**Note the GCS URI for your data** After uploading, click on a file and find its **gs:// URI** (e.g., `gs://johndoe-titanic/titanic_test.csv`). This URI will be used to access the data in your notebook.

## Adjust bucket permissions

We need to grant the Compute Engine default service account IAM roles so that our Workbench notebooks and training jobs can interact with the bucket. Open **Cloud Shell** — a browser-based terminal built into the Google Cloud Console (click the terminal icon **>\_** in the top-right toolbar).

#### Find your service account

```sh
gcloud iam service-accounts list --filter="displayName:Compute Engine default service account" --format="value(email)"
```

This will return an email like `123456789-compute@developer.gserviceaccount.com`. Use that value in the commands below.

<!-- shared workshop service account: 549047673858-compute@developer.gserviceaccount.com -->

#### Grant permissions

Replace `YOUR_BUCKET_NAME` and `YOUR_SERVICE_ACCOUNT`, then run:

```sh
# Grant read permissions on the bucket
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET_NAME \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
  --role="roles/storage.objectViewer"

# Grant write permissions on the bucket
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET_NAME \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
  --role="roles/storage.objectCreator"

# (Only if you also need overwrite/delete)
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET_NAME \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
  --role="roles/storage.objectAdmin"
```

<!-- prefilled example:
gcloud storage buckets add-iam-policy-binding gs://johndoe-titanic \
  --member="serviceAccount:549047673858-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
-->

::::::::::::::::::::::::::::::::::::: callout

#### `gcloud storage` vs. `gsutil`
Older tutorials often reference `gsutil` for Cloud Storage operations. Google now recommends `gcloud storage` as the primary CLI. Both work, but `gcloud storage` is actively maintained and consistent with the rest of the `gcloud` CLI.

::::::::::::::::::::::::::::::::::::::::::::::::

## Data transfer & storage costs

GCS costs are based on storage class, data transfer, and operations (requests).

- **Standard storage**: ~$0.02 per GB per month in `us-central1`.
- **Uploading data (ingress)**: Free.
- **Downloading data out of GCP (egress)**: ~$0.12 per GB.
- **Cross-region access**: ~$0.01–0.02 per GB within North America.
- **GET requests**: ~$0.004 per 10,000 requests.
- **PUT/POST requests**: ~$0.05 per 10,000 requests.
- **Deleting data**: Free (but Nearline/Coldline/Archive early-deletion fees apply).

***For detailed pricing, see [GCS Pricing Information](https://cloud.google.com/storage/pricing).***

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Estimating Storage Costs

**1. Estimate the total cost of storing 1 GB in GCS Standard storage (us-central1) for one month assuming:**
- Dataset retrieved 100 times for training and tuning
- Data is downloaded once out of GCP at the end of the project

**2. Repeat the above calculation for datasets of 10 GB, 100 GB, and 1 TB (1024 GB).**

**Hints**: Storage $0.02/GB/month, Egress $0.12/GB, GET requests negligible at this scale.

:::::::::::::::: solution

1. **1 GB**: Storage $0.02 + Egress $0.12 = **$0.14**
2. **10 GB**: $0.20 + $1.20 = **$1.40**
3. **100 GB**: $2.00 + $12.00 = **$14.00**
4. **1 TB**: $20.48 + $122.88 = **$143.36**

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::


## Accessing data from your notebook

Now that our bucket is set up, let's use it from the Workbench notebook you created in the previous episode.

If you haven't already cloned the repository, open JupyterLab from your Workbench Instance and run `!git clone https://github.com/qualiaMachine/Intro_GCP_for_ML.git`. Then navigate to `/Intro_GCP_for_ML/notebooks/03-Data-storage-and-access.ipynb`.

### Set up GCP environment
Before interacting with GCS, initialize the client libraries. The `storage.Client()` call creates a connection using the credentials already attached to your Workbench VM.

```python
from google.cloud import storage
client = storage.Client()
print("Project:", client.project)
```

### Reading data directly into memory

```python
import pandas as pd
import io

bucket_name = "johndoe-titanic" # ADJUST to your bucket's name

bucket = client.bucket(bucket_name)
blob = bucket.blob("titanic_train.csv")
train_data = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
print(train_data.shape)
train_data.head()
```

The Titanic dataset contains passenger information (age, class, fare, etc.) and a binary survival label — we'll train a classifier on this data in Episode 4.

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

- **`Forbidden (403)`** — Your service account lacks permission. Revisit the **Adjust bucket permissions** section above.
- **`NotFound (404)`** — The bucket name or file path is wrong. Double-check `bucket_name` and the blob path with `client.list_blobs(bucket_name)`.
- **`DefaultCredentialsError`** — The notebook cannot find credentials. Make sure you are running on a Vertex AI Workbench Instance (not a local machine).

:::::::::::::::::::::::::::::::::::::

## Monitoring storage usage and costs

It's good practice to periodically check how much storage your bucket is using. The code below sums up all object sizes.

```python
total_size_bytes = 0
bucket = client.bucket(bucket_name)

for blob in client.list_blobs(bucket_name):
    total_size_bytes += blob.size

total_size_mb = total_size_bytes / (1024**2)
print(f"Total size of bucket '{bucket_name}': {total_size_mb:.2f} MB")
```

```python
storage_price_per_gb = 0.02   # $/GB/month for Standard storage
egress_price_per_gb = 0.12    # $/GB for internet egress (same-region transfers are free)
total_size_gb = total_size_bytes / (1024**3)

monthly_storage = total_size_gb * storage_price_per_gb
egress_cost = total_size_gb * egress_price_per_gb

print(f"Bucket size: {total_size_gb:.4f} GB")
print(f"Estimated monthly storage cost: ${monthly_storage:.4f}")
print(f"Estimated annual storage cost:  ${monthly_storage*12:.4f}")
print(f"One-time full download (egress) cost: ${egress_cost:.4f}")
```

## Writing output files to GCS

```python
# Create a sample file locally on the notebook VM
file_path = "/home/jupyter/Notes.txt"
with open(file_path, "w") as f:
    f.write("This is a test note for GCS.")
```

```python
bucket = client.bucket(bucket_name)
blob = bucket.blob("docs/Notes.txt")
blob.upload_from_filename(file_path)
print("File uploaded successfully.")
```

List bucket contents:

```python
for blob in client.list_blobs(bucket_name):
    print(blob.name)
```

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Read and explore the test dataset

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

### Challenge 3: Upload a summary CSV to GCS

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


## Removing unused data (complete *after* the workshop)

After you are done using your data, remove unused files/buckets to stop costs:

- **Option 1: Delete files only** – In your bucket, select the files you want to remove and click **Delete**.
- **Option 2: Delete the bucket entirely** – In **Cloud Storage > Buckets**, select your bucket and click **Delete**.

For a detailed walkthrough of cleaning up all workshop resources, see [Episode 9: Resource Management and Cleanup](09-Resource-management-cleanup.md).

::::::::::::::::::::::::::::::::::::: keypoints

- Use GCS for scalable, cost-effective, and persistent storage in GCP.
- Persistent disks are suitable only for small, temporary datasets.
- Load data from GCS into memory with `storage.Client` or directly via `pd.read_csv("gs://...")`.
- Periodically check storage usage and estimate costs to manage your GCS budget.
- Track your storage, transfer, and request costs to manage expenses.
- Regularly delete unused data or buckets to avoid ongoing costs.

::::::::::::::::::::::::::::::::::::::::::::::::
