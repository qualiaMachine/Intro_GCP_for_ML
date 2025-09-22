---
title: "Data Storage: Setting up GCS"
teaching: 15
exercises: 5
---

:::::::::::::::::::::::::::::::::::::: questions

- How can I store and manage data effectively in GCP for Vertex AI workflows?  
- What are the advantages of Google Cloud Storage (GCS) compared to local or VM storage for machine learning projects?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain data storage options in GCP for machine learning projects.  
- Describe the advantages of GCS for large datasets and collaborative workflows.  
- Outline steps to set up a GCS bucket and manage data within Vertex AI.  

::::::::::::::::::::::::::::::::::::::::::::::::

## Storing data on GCP
Machine learning and AI projects rely on data, making efficient storage and management essential. Google Cloud offers several storage options, but the most common for ML workflows are **persistent disks** (attached to Compute Engine VMs or Vertex AI Workbench) and **Google Cloud Storage (GCS) buckets**.  

> #### Consult your institution's IT before handling sensitive data in GCP
> As with AWS, **do not upload restricted or sensitive data to GCP services unless explicitly approved by your institution’s IT or cloud security team**. For regulated datasets (HIPAA, FERPA, proprietary), work with your institution to ensure encryption, restricted access, and compliance with policies.

## Options for storage: VM Disks or GCS

### What is a VM persistent disk?
A persistent disk is the storage volume attached to a Compute Engine VM or a Vertex AI Workbench notebook. It can store datasets and intermediate results, but it is tied to the lifecycle of the VM.  

### When to store data directly on a persistent disk
- Useful for small, temporary datasets processed interactively.  
- Data persists if the VM is stopped, but storage costs continue as long as the disk exists.  
- Not ideal for collaboration, scaling, or long-term dataset storage.  

::::::::::::::::::::::::::::::::::::: callout 

### Limitations of persistent disk storage
- **Scalability**: Limited by disk size quota.  
- **Sharing**: Harder to share across projects or team members.  
- **Cost**: More expensive per GB compared to GCS for long-term storage.  

::::::::::::::::::::::::::::::::::::::::::::::::

### What is a GCS bucket?
For most ML workflows in Vertex AI, **Google Cloud Storage (GCS) buckets** are recommended. A GCS bucket is a container in Google’s object storage service where you can store an essentially unlimited number of files. Data in GCS can be accessed from Vertex AI training jobs, Workbench notebooks, and other GCP services using a **GCS URI** (e.g., `gs://your-bucket-name/your-file.csv`).  

::::::::::::::::::::::::::::::::::::: callout 

### Benefits of using GCS (recommended for ML workflows)
- **Separation of storage and compute**: Data remains available even if VMs or notebooks are deleted.  
- **Easy sharing**: Buckets can be accessed by collaborators with the right IAM roles.  
- **Integration with Vertex AI and BigQuery**: Read and write data directly in pipelines.  
- **Scalability**: Handles datasets of any size without disk limits.  
- **Cost efficiency**: Lower cost than persistent disks for long-term storage.  
- **Data persistence**: Durable and highly available across regions.  

::::::::::::::::::::::::::::::::::::::::::::::::

## Recommended approach: GCS buckets

To upload our Titanic dataset to a GCS bucket, we’ll follow these steps:

1. Log in to the Google Cloud Console.  
2. Create a new bucket (or use an existing one).  
3. Upload your dataset files.  
4. Use the GCS URI to reference your data in Vertex AI workflows.  

### Detailed procedure

##### 1. Sign in to Google Cloud Console
- Go to [console.cloud.google.com](https://console.cloud.google.com) and log in with your credentials.  

##### 2. Navigate to Cloud Storage
- In the search bar, type **Storage**.  
- Click **Cloud Storage > Buckets**.  

##### 3. Create a new bucket
- Click **Create bucket**.  
- Enter a globally unique name (e.g., `yourname-titanic-gcs`).
- **Labels (tags)**: Add labels to track resource usage and billing
    - `purpose=workshop`
    - `owner=lastname_firstname`   
- Choose a location type:  
  - **Region** (cheapest, good default).  
  - **Multi-region** (higher redundancy, more expensive).  
- **Access Control**: Recommended: Uniform access with IAM.  
- **Public Access**: Block public access unless explicitly needed.  
- **Versioning**: Disable unless you want to keep multiple versions of files.  

##### 4. Set bucket permissions
- By default, only project members can access.  
- To grant Vertex AI service accounts access, assign the `Storage Object Admin` or `Storage Object Viewer` role at the bucket level.  

##### 5. Upload files to the bucket
- If you haven’t downloaded them yet, right-click and save as `.csv`:  
  - [titanic_train.csv](https://raw.githubusercontent.com/UW-Madison-DataScience/ml-with-aws-sagemaker/main/data/titanic_train.csv)  
  - [titanic_test.csv](https://raw.githubusercontent.com/UW-Madison-DataScience/ml-with-aws-sagemaker/main/data/titanic_test.csv)  
- In the bucket dashboard, click **Upload Files**.  
- Select your Titanic CSVs and upload.  

##### 6. Note the GCS URI for your data
- After uploading, click on a file and find its **gs:// URI** (e.g., `gs://yourname-titanic-gcs/titanic_train.csv`).  
- This URI will be used when launching Vertex AI training jobs.  

## GCS bucket costs

GCS costs are based on storage class, data transfer, and operations (requests).  

### Storage costs
- Standard storage (us-central1): ~$0.02 per GB per month.  
- Other classes (Nearline, Coldline, Archive) are cheaper but with retrieval costs.  

### Data transfer costs
- Uploading data into GCS is free.  
- Downloading data out of GCP costs ~$0.12 per GB.  
- Accessing data within the same region is free.  

### Request costs
- `GET` (read) requests: ~$0.004 per 10,000 requests.  
- `PUT` (write) requests: ~$0.05 per 10,000 requests.  

***For detailed pricing, see [GCS Pricing Information](https://cloud.google.com/storage/pricing).***  

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge: Estimating Storage Costs

**1. Estimate the total cost of storing 1 GB in GCS Standard storage (us-central1) for one month assuming:**  
- Storage duration: 1 month  
- Dataset retrieved 100 times for model training and tuning  
- Data is downloaded once out of GCP at the end of the project  

**Hints**  
- Storage cost: $0.02 per GB per month  
- Egress (download out of GCP): $0.12 per GB  
- `GET` requests: $0.004 per 10,000 requests (100 requests ≈ free for our purposes)  

**2. Repeat the above calculation for datasets of 10 GB, 100 GB, and 1 TB (1024 GB).**

:::::::::::::::: solution

1. **1 GB**:  
- Storage: 1 GB × $0.02 = $0.02  
- Egress: 1 GB × $0.12 = $0.12  
- Requests: ~0 (100 reads well below pricing tier)  
- **Total: $0.14**

1. **10 GB**:  
- Storage: 10 GB × $0.02 = $0.20  
- Egress: 10 GB × $0.12 = $1.20  
- Requests: ~0  
- **Total: $1.40**

1. **100 GB**:  
- Storage: 100 GB × $0.02 = $2.00  
- Egress: 100 GB × $0.12 = $12.00  
- Requests: ~0  
- **Total: $14.00**

1. **1 TB (1024 GB)**:  
- Storage: 1024 GB × $0.02 = $20.48  
- Egress: 1024 GB × $0.12 = $122.88  
- Requests: ~0  
- **Total: $143.36**

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Removing unused data (complete *after* the workshop)

After you are done using your data, remove unused files/buckets to stop costs:  

- **Option 1: Delete files only** – if you plan to reuse the bucket.  
- **Option 2: Delete the bucket entirely** – if you no longer need it.  

::::::::::::::::::::::::::::::::::::: keypoints 

- Use GCS for scalable, cost-effective, and persistent storage in GCP.  
- Persistent disks are suitable only for small, temporary datasets.  
- Track your storage, transfer, and request costs to manage expenses.  
- Regularly delete unused data or buckets to avoid ongoing costs.  

::::::::::::::::::::::::::::::::::::::::::::::::
