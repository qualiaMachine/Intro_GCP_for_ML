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
Machine learning and AI projects rely on data, making efficient storage and management essential. Google Cloud offers several storage options, but the most common for ML workflows are **Virtual Machine (VM) disks** and **Google Cloud Storage (GCS) buckets**.  

> #### Consult your institution's IT before handling sensitive data in GCP
> As with AWS, **do not upload restricted or sensitive data to GCP services unless explicitly approved by your institution's IT or cloud security team**. For regulated datasets (HIPAA, FERPA, proprietary), work with your institution to ensure encryption, restricted access, and compliance with policies.

## Options for storage: VM Disks or GCS

### What is a VM  disk?
A VM disk is the storage volume attached to a Compute Engine VM or a Vertex AI Workbench notebook. It can store datasets and intermediate results, but it is tied to the lifecycle of the VM.  

### When to store data directly on a VM disk
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
For most ML workflows in GCP, **Google Cloud Storage (GCS) buckets** are recommended. A GCS bucket is a container in Google's object storage service where you can store an essentially unlimited number of files. Data in GCS can be accessed from Vertex AI training jobs, Workbench notebooks, and other GCP services using a *GCS URI* (e.g., `gs://your-bucket-name/your-file.csv`).  

::::::::::::::::::::::::::::::::::::: callout 

### Benefits of using GCS (recommended for ML workflows)
- **Separation of storage and compute**: Data remains available even if VMs or notebooks are deleted.  
- **Easy sharing**: Buckets can be accessed by collaborators with the right IAM roles.  
- **Integration with Vertex AI and BigQuery**: Read and write data directly using other GCP tools.  
- **Scalability**: Handles datasets of any size without disk limits.  
- **Cost efficiency**: Lower cost than persistent disks for long-term storage.  
- **Data persistence**: Durable and highly available across regions.  

::::::::::::::::::::::::::::::::::::::::::::::::

## Recommended approach: GCS buckets

To upload our Titanic dataset to a GCS bucket, we'll follow these steps:

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
- **A) Provide a bucket name**: Enter a globally unique name. For this workshop, we can use the following naming convention to easily locate our buckets: `teamname_first-lastname_titanic` (e.g., sinkorswim_john-doe_titanic)
- **B) Add Labels (tags)**: Add labels to track resource usage and billing. If you're working in a shared account, this step is *mandatory*. If not, it's still recommended to help you track your own costs!
    - `project=teamname`
    - `name=first-lastname`
    - `purpose=bucket-dataname`
 
![Example of Tags for a GCP Bucket](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/images/bucket-tags.jpg){alt="Screenshot showing required tags for a GCP bucket"}

- **Choose where to store your data**:  When creating a storage bucket in Google Cloud, the best practice for most machine learning workflows is to use a regional bucket in the same region as your compute resources (for example, us-central1). This setup provides the lowest latency and avoids network egress charges when training jobs read from storage, while also keeping costs predictable. A multi-region bucket, on the other hand, can make sense if your primary goal is broad availability or if collaborators in different regions need reliable access to the same data; the trade-off is higher cost and the possibility of extra egress charges when pulling data into a specific compute region. For most research projects, a regional bucket with the Standard storage class, uniform access control, and public access prevention enabled offers a good balance of performance, security, and affordability.
  - **Region** (cheapest, good default). For instance, us-central1 (Iowa) costs $0.020 per GB-month.
  - **Multi-region** (higher redundancy, more expensive).  
- **Choose how to store your data (storage class)**: When creating a bucket, you'll be asked to choose a storage class, which determines how much you pay for storing data and how often you're allowed to access it without extra fees.
    - **Standard** – best for active ML/AI workflows. Training data is read and written often, so this is the safest default.
    - **Nearline / Coldline / Archive** – designed for backups or rarely accessed files. These cost less per GB to store, but you pay retrieval fees if you read them during training. Not recommended for most ML projects where data access is frequent.
    - You may see an option to "Enable hierarchical namespace". GCP now offers an option to enable a hierarchical namespace for buckets, but this is mainly useful for large-scale analytics pipelines. For most ML workflows, the standard flat namespace is simpler and fully compatible—so it's best to leave this option off.
      
- **Choose how to control access to objects:**  
  For ML projects, you should **prevent public access** so that only authorized users can read or write data. This keeps research datasets private and avoids accidental exposure.
  
  When prompted to choose an access control method, choose **uniform access** unless you have a very specific reason to manage object-level permissions.
  - **Uniform access (recommended):** Simplifies management by enforcing permissions at the bucket level using IAM roles. It's the safer and more maintainable choice for teams and becomes permanent after 90 days.  
  - **Fine-grained access:** Allows per-file permissions using ACLs, but adds complexity and is rarely needed outside of web hosting or mixed-access datasets.
    

##### 4. Upload files to the bucket
- If you haven't downloaded them yet, right-click and save as `.csv`:  
  - [titanic_train.csv](https://raw.githubusercontent.com/UW-Madison-DataScience/ml-with-aws-sagemaker/main/data/titanic_train.csv)  
  - [titanic_test.csv](https://raw.githubusercontent.com/UW-Madison-DataScience/ml-with-aws-sagemaker/main/data/titanic_test.csv)  
- In the bucket dashboard, click **Upload Files**.  
- Select your Titanic CSVs and upload.  

**Note the GCS URI for your data** After uploading, click on a file and find its **gs:// URI** (e.g., `gs://yourname-titanic-gcs/titanic_train.csv`). This URI will be used to access the data later.

## GCS bucket costs

GCS costs are based on storage class, data transfer, and operations (requests).  

### Storage costs
- Standard storage (us-central1): ~$0.02 per GB per month.  
- Other classes (Nearline, Coldline, Archive) are cheaper but with retrieval costs.  

### Data transfer costs explained  
- **Uploading data (ingress):** Copying data into a GCS bucket from your laptop, campus HPC, or another provider is free.  
- **Accessing data in the same region:** If your bucket and your compute resources (VMs, Vertex AI jobs) are in the same region, you can read and stream data with no transfer fees. You only pay the storage cost per GB-month.  
- **Cross-region access:** If your bucket is in one region and your compute runs in another, you'll pay an egress fee (about $0.01–0.02 per GB within North America, higher if crossing continents).  
- **Downloading data out of GCP (egress):** This refers to data leaving Google's network to the public internet, such as downloading files to your laptop. Typical cost is around $0.12 per GB to the U.S. and North America, more for other continents.  
- **Deleting data:** Removing objects or buckets does not incur transfer costs. If you download data before deleting, you pay for the egress, but simply deleting in the console or CLI is free. For Nearline/Coldline/Archive storage classes, deleting before the minimum storage duration (30, 90, or 365 days) triggers an early deletion fee.  

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

## When does BigQuery come into play?

For many ML workflows, especially smaller projects or those centered on image, text, or modest tabular datasets, BigQuery is overkill. GCS buckets are usually enough to store and access your data for training jobs. That said, BigQuery can be valuable when you are working with large tabular datasets and need a shared environment for exploration or collaboration. Instead of every team member downloading the same CSVs, BigQuery lets everyone query the data in place with SQL, share results through saved queries or views, and control access at the dataset or table level with IAM. BigQuery also integrates with Vertex AI, so if your data is already structured and stored there, you can connect it directly to training pipelines. The trade-off is cost: you pay not only for storage but also for the amount of data scanned by queries. For many ML research projects this is unnecessary, but when teams need a centralized, queryable workspace for large tabular data, BigQuery can simplify collaboration.  

::::::::::::::::::::::::::::::::::::: keypoints 

- Use GCS for scalable, cost-effective, and persistent storage in GCP.  
- Persistent disks are suitable only for small, temporary datasets.  
- Track your storage, transfer, and request costs to manage expenses.  
- Regularly delete unused data or buckets to avoid ongoing costs.  

::::::::::::::::::::::::::::::::::::::::::::::::
