---
title: "Resource Management & Monitoring on Vertex AI (GCP)"
teaching: 30
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions

- How do I monitor and control Vertex AI, Workbench, and GCS costs day‑to‑day?
- What *specifically* should I stop, delete, or schedule to avoid surprise charges?
- How do I set budget alerts so cost leaks get caught quickly?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Identify the major cost drivers across Vertex AI (training jobs, endpoints, Workbench notebooks) and GCS, with ballpark costs.
- Practice safe cleanup for Workbench Instances, training/tuning jobs, batch predictions, models, and endpoints.
- Set a budget alert and apply labels to keep costs visible and predictable.
- Use `gcloud` commands for auditing and rapid cleanup.

::::::::::::::::::::::::::::::::::::::::::::::::

You've now run training jobs, tuning jobs, and possibly deployed models across the previous episodes. Before moving on, let's make sure none of those resources are still billing you — and learn the habits that prevent surprise charges going forward.

::::::::::::::::::::::::::::::::::::: callout

### Continuing to Episodes 10–11?

If you plan to work through **Episode 10 (RAG)** or **Episode 11 (CLI Workflows)** after this one, **keep your Workbench Instance and GCS bucket** — just make sure to **stop the notebook runtime** when you're not actively using it. You can come back to the full teardown checklist at the very end of the workshop.

::::::::::::::::::::::::::::::::::::::::::::::::

## Check your current spend first

Before cleaning anything up, find out where you stand. Open the **Cloud Console** and navigate to:

**Billing → Reports**

- Set the time range to **This month** (or **Today** for workshop use).
- Group by **Service** to see which GCP services are costing the most.
- Look for **Compute Engine** (backs Workbench VMs and training jobs), **Vertex AI**, and **Cloud Storage**.

This is the single most important dashboard to bookmark. If you only learn one thing from this episode, it's where to find this page.

You can also check from the CLI:

```bash
# Quick check: is my project accumulating Vertex AI resources right now?
gcloud ai endpoints list --region=us-central1
gcloud workbench instances list --location=us-central1-a
gcloud ai custom-jobs list --region=us-central1 --filter="state=JOB_STATE_RUNNING"
```


## What costs you money on GCP (and how much)

Not all resources cost equally. Here are the main cost drivers you'll encounter in this workshop, ordered from most to least dangerous:

| Resource | Billing model | Ballpark cost | Risk level |
|----------|--------------|---------------|------------|
| **Vertex AI endpoints** | Per node‑hour, **24/7 while deployed** | ~$4.50/day for one `n1-standard-4` node | **High** — bills even with zero traffic |
| **Workbench Instances** (running) | Per VM‑hour + GPU | ~$0.19/hr CPU‑only (`n1-standard-4`); add ~$0.35/hr per T4 GPU | **High** — easy to forget overnight |
| **Training / HPT jobs** | Per VM/GPU‑hour while running | Same VM rates; auto‑stops when done | **Medium** — usually self‑limiting |
| **Workbench disks** (stopped VM) | Per GB‑month for persistent disk | ~$0.04/GB/month (~$4/month for 100 GB) | **Low** — small but adds up |
| **GCS storage** | Per GB‑month + operations + egress | ~$0.02/GB/month (Standard) | **Low** — cheap until multi‑TB |
| **Network egress** | Per GB downloaded out of GCP | ~$0.12/GB | **Low** — avoid large downloads to local |

> **Rule of thumb:** Endpoints left deployed and notebooks left running are the most common surprise bills in education and research settings.


## Shutting down Workbench Instances

In Episode 3 we created a **Workbench Instance** — the currently recommended notebook environment. Here's how to stop or delete it:

### Stop via Console
Vertex AI → **Workbench** → **Instances** tab → select your instance → **Stop**.

### Stop via CLI
```bash
# List all Workbench Instances in your zone
gcloud workbench instances list --location=us-central1-a

# Stop an instance (stops VM billing; disk charges continue)
gcloud workbench instances stop INSTANCE_NAME --location=us-central1-a
```

### Delete when you're done for good
```bash
# Permanently delete the instance and its disk
gcloud workbench instances delete INSTANCE_NAME --location=us-central1-a --quiet
```

### Enable idle shutdown (recommended)
You can configure your instance to auto‑stop after a period of inactivity, so you never accidentally leave it running overnight:

- **Console**: Select your instance → **Edit** → set **Idle shutdown** to 60–120 minutes.
- **At creation time**: Add `--idle-shutdown-timeout=60` to your `gcloud workbench instances create` command.

> **Disks still cost money while the VM is stopped** (~$4/month for 100 GB). If you're completely done with an instance, **delete** it rather than just stopping it.


## Cleaning up training, tuning, and batch jobs

Training and HPT jobs automatically stop billing when they finish, but it's good practice to audit for jobs stuck in `RUNNING` and to delete old jobs you no longer need.

### Audit with CLI
```bash
# Custom training jobs
gcloud ai custom-jobs list --region=us-central1

# Hyperparameter tuning jobs
gcloud ai hp-tuning-jobs list --region=us-central1

# Batch prediction jobs
gcloud ai batch-prediction-jobs list --region=us-central1
```

Each command prints a table showing the job ID, display name, state (e.g., `JOB_STATE_SUCCEEDED`, `JOB_STATE_RUNNING`), and creation time. Look for any jobs stuck in `RUNNING` — those are still consuming resources.

### Cancel or delete as needed
```bash
# Cancel a running job
gcloud ai custom-jobs cancel JOB_ID --region=us-central1

# Delete a completed job you no longer need
gcloud ai custom-jobs delete JOB_ID --region=us-central1
```

> **Tip:** Keep one "golden" successful job per experiment for reference, then delete the rest to reduce console clutter.


## Undeploy models and delete endpoints (major cost pitfall)

Deployed endpoints are billed per node‑hour **24/7**, even with zero prediction traffic. A single forgotten endpoint can cost ~$135/month. Always undeploy models before deleting the endpoint.

### Find endpoints and deployed models
```bash
gcloud ai endpoints list --region=us-central1
gcloud ai endpoints describe ENDPOINT_ID --region=us-central1
```

### Undeploy and delete
```bash
# Step 1: Undeploy the model (stops node-hour billing)
gcloud ai endpoints undeploy-model ENDPOINT_ID \
  --deployed-model-id=DEPLOYED_MODEL_ID \
  --region=us-central1 \
  --quiet

# Step 2: Delete the endpoint itself
gcloud ai endpoints delete ENDPOINT_ID \
  --region=us-central1 \
  --quiet
```

> **Model Registry note:** Keeping a model *registered* (but not deployed to an endpoint) does not incur node‑hour charges. You only pay a small amount for the model artifact storage in GCS.


## GCS housekeeping

### Check bucket size
```bash
# Human-readable bucket size
gcloud storage du gs://YOUR_BUCKET --summarize --readable-sizes

# List top-level contents
gcloud storage ls gs://YOUR_BUCKET
```

> **Note:** `gsutil` commands (e.g., `gsutil du`, `gsutil ls`) still work but are being replaced by `gcloud storage`. We use the newer syntax here.

### Lifecycle policies
A lifecycle policy tells GCS to automatically delete or transition objects based on rules you define. This is useful for cleaning up temporary training outputs.

Save the following as `lifecycle.json`:
```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 7, "matchesPrefix": ["tmp/"]}
      },
      {
        "action": {"type": "Delete"},
        "condition": {"numNewerVersions": 3}
      }
    ]
  }
}
```

- **Rule 1**: Auto‑delete any object under `tmp/` that is older than 7 days.
- **Rule 2**: If versioning is enabled, keep only the 3 most recent versions.

Apply it:
```bash
gcloud storage buckets update gs://YOUR_BUCKET --lifecycle-file=lifecycle.json

# Verify
gcloud storage buckets describe gs://YOUR_BUCKET --format="yaml(lifecycle)"
```

### Egress reminder
Downloading data out of GCP to your laptop costs ~$0.12/GB. Prefer **in‑cloud** training and evaluation, and share results via GCS links rather than local downloads.


## Labels and budgets

### Standardize labels on all resources
Labels let you track costs per user, team, or experiment in billing reports. Apply them consistently:

- Examples: `owner=yourname`, `team=ml-workshop`, `purpose=titanic-demo`
- The Vertex AI Python SDK supports labels on job creation; `gcloud` commands accept `--labels=key=value,...`

### Set budget alerts (do this now)
This is the single most protective action you can take:

1. Go to **Billing → Budgets & alerts** in the Cloud Console.
2. Click **Create budget**.
3. Set a budget amount (e.g., $10 or $25 for a workshop).
4. Set alert thresholds at **50%**, **80%**, and **100%**.
5. Add **forecast‑based alerts** to catch trends before you hit the limit.
6. Make sure email notifications go to **all project maintainers**, not just you.

> **For production use:** You can export detailed billing data to BigQuery for cost analysis by service, label, or SKU. See the [billing export documentation](https://cloud.google.com/billing/docs/how-to/export-data-bigquery) for setup instructions.


## Common pitfalls and quick fixes

| Pitfall | Fix |
|---------|-----|
| Forgotten endpoint billing 24/7 | Undeploy models → delete endpoint |
| Notebook left running over weekend | Enable **idle shutdown** (60–120 min) |
| Duplicate datasets across buckets | Consolidate to one bucket; set lifecycle to purge `tmp/` |
| Too many parallel HPT trials | Cap `parallel_trial_count` to 2–4 |
| Don't know what's costing money | Check **Billing → Reports**; add labels to all resources |

::::::::::::::::::::::::::::::::::::: callout

### Going further: automating cleanup

Once you move from workshop use to regular research, consider automating resource cleanup:

- **Cloud Scheduler** can run a nightly job to stop idle Workbench Instances via the Vertex AI API.
- **Cloud Functions** or **Cloud Run** can periodically sweep for forgotten endpoints.
- **Budget alerts** can trigger Pub/Sub messages that automatically shut down resources when spend exceeds a threshold.

These are beyond the scope of this workshop, but the [Cloud Scheduler documentation](https://cloud.google.com/scheduler/docs) is a good starting point.

::::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1 — Check your spend and set a budget

1. Navigate to **Billing → Reports** in the Cloud Console. Find your project's current‑month spend grouped by service.
2. Navigate to **Billing → Budgets & alerts**. Create a **$10 budget** with alert thresholds at 50% and 100%.

:::::::::::::::: solution

1. In the Cloud Console, click the **Navigation menu (☰)** → **Billing** → **Reports**. Set time range to "This month" and group by "Service." You should see Compute Engine, Vertex AI, and Cloud Storage if you've been running workshop exercises.

2. Go to **Billing** → **Budgets & alerts** → **Create budget**. Set:
   - **Name**: `workshop-budget`
   - **Amount**: `$10`
   - **Thresholds**: 50% ($5) and 100% ($10)
   - **Alerts to**: your email address

Click **Finish** to activate the budget.

::::::::::::::::

::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2 — Find and stop idle notebooks

List all running Workbench Instances in your zone and stop any you are not actively using.

```bash
gcloud workbench instances list --location=us-central1-a
```

:::::::::::::::: solution

```bash
# List instances — look for STATE=ACTIVE
gcloud workbench instances list --location=us-central1-a

# Stop an instance you're not using
gcloud workbench instances stop INSTANCE_NAME --location=us-central1-a
```

If the instance shows `STATE=ACTIVE` and you're not currently working in it, stop it. You can restart it later with `gcloud workbench instances start`.

::::::::::::::::

::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 3 — Endpoint sweep

List all deployed endpoints in your region, undeploy any model you don't need, and delete the endpoint.

:::::::::::::::: solution

```bash
# List all endpoints
gcloud ai endpoints list --region=us-central1

# Pick an endpoint ID from the list, then inspect it
gcloud ai endpoints describe ENDPOINT_ID --region=us-central1

# Undeploy the model (find DEPLOYED_MODEL_ID in the describe output)
gcloud ai endpoints undeploy-model ENDPOINT_ID \
  --deployed-model-id=DEPLOYED_MODEL_ID \
  --region=us-central1 \
  --quiet

# Delete the now-empty endpoint
gcloud ai endpoints delete ENDPOINT_ID \
  --region=us-central1 \
  --quiet
```

::::::::::::::::

::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 4 — Write and apply a lifecycle policy

Create a GCS lifecycle rule that deletes objects under `tmp/` after 7 days and keeps only 3 versions of versioned objects. Apply it to your bucket.

:::::::::::::::: solution

Save the following as `lifecycle.json`:
```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 7, "matchesPrefix": ["tmp/"]}
      },
      {
        "action": {"type": "Delete"},
        "condition": {"numNewerVersions": 3}
      }
    ]
  }
}
```

Apply and verify:
```bash
gcloud storage buckets update gs://YOUR_BUCKET --lifecycle-file=lifecycle.json
gcloud storage buckets describe gs://YOUR_BUCKET --format="yaml(lifecycle)"
```

::::::::::::::::

::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 5 — Full workshop teardown

If you are done with all episodes, perform a complete cleanup:

1. Stop or delete your Workbench Instance.
2. Verify no endpoints are deployed.
3. Delete any completed training/tuning jobs you don't need.
4. Check your GCS bucket — remove any files you don't want to keep, or delete the bucket entirely.

:::::::::::::::: solution

```bash
# 1. Delete your Workbench Instance
gcloud workbench instances delete INSTANCE_NAME \
  --location=us-central1-a --quiet

# 2. Confirm no endpoints remain
gcloud ai endpoints list --region=us-central1
# (If any appear, undeploy models and delete them as shown above)

# 3. Delete old training jobs
gcloud ai custom-jobs list --region=us-central1
gcloud ai custom-jobs delete JOB_ID --region=us-central1

gcloud ai hp-tuning-jobs list --region=us-central1
gcloud ai hp-tuning-jobs delete JOB_ID --region=us-central1

# 4. Remove your GCS bucket (WARNING: this deletes all data in the bucket)
gcloud storage rm -r gs://YOUR_BUCKET
```

After cleanup, check **Billing → Reports** one more time to confirm no services are still accumulating charges.

::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::


## End‑of‑session checklist

Before you close your laptop, run through this quick checklist:

1. **Workbench Instances** — stopped (or deleted if you're done for good).
2. **Training / HPT jobs** — no jobs stuck in `RUNNING`.
3. **Endpoints** — all models undeployed; unused endpoints deleted.
4. **GCS** — no large temporary files lingering; lifecycle policy in place.
5. **Budget alert** — set and sending to your email.

> Bookmark **Billing → Reports** and check it at the start of each session. A 10‑second glance can save you from a surprise bill.

::::::::::::::::::::::::::::::::::::: keypoints

- **Check Billing → Reports** regularly — know what you're spending before it surprises you.
- **Endpoints** and **running notebooks** are the most common cost leaks; undeploy and stop first.
- **Set a budget alert** — it's the single most protective action you can take.
- Configure **idle shutdown** on Workbench Instances so forgotten notebooks auto‑stop.
- Keep storage tidy with **GCS lifecycle policies** and avoid duplicate datasets.
- Use **labels** on all resources so you can trace costs in billing reports.

::::::::::::::::::::::::::::::::::::::::::::::::
