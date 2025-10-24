---
title: "Resource Management & Monitoring on Vertex AI (GCP)"
teaching: 45
exercises: 20
---

:::::::::::::::::::::::::::::::::::::: questions

- How do I monitor and control Vertex AI, Workbench, and GCS costs day‑to‑day?
- What *specifically* should I stop, delete, or schedule to avoid surprise charges?
- How can I automate cleanup and set alerting so leaks get caught quickly?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Identify all major cost drivers across Vertex AI (training jobs, endpoints, Workbench notebooks, batch prediction) and GCS.
- Practice safe cleanup for **Managed** and **User‑Managed** Workbench notebooks, training/tuning jobs, batch predictions, models, endpoints, and artifacts.
- Configure budgets, labels, and basic lifecycle policies to keep costs predictable.
- Use `gcloud`/`gsutil` commands for auditing and rapid cleanup; understand when to prefer the Console.
- Draft simple automation patterns (Cloud Scheduler + `gcloud`) to enforce idle shutdown.

::::::::::::::::::::::::::::::::::::::::::::::::

## What costs you money on GCP (quick map)

- **Vertex AI training jobs** (Custom Jobs, Hyperparameter Tuning Jobs) — billed per VM/GPU hour while running.
- **Vertex AI endpoints (online prediction)** — billed per node‑hour *24/7 while deployed*, even if idle.
- **Vertex AI batch prediction jobs** — billed for the job’s compute while running.
- **Vertex AI Workbench notebooks** — the backing VM and disk bill while running (and disks bill even when stopped).
- **GCS buckets** — storage class, object count/size, versioning, egress, and request ops.
- **Artifact Registry** (containers, models) — storage for images and large artifacts.
- **Network egress** — downloading data out of GCP (e.g., to your laptop) incurs cost.
- **Logging/Monitoring** — high‑volume logs/metrics can add up (rare in small workshops, real in prod).

> Rule of thumb: **Endpoints left deployed** and **notebooks left running** are the most common surprise bills in education/research settings.

## A daily “shutdown checklist” (use now, automate later)

1) **Workbench notebooks** — stop the runtime/instance when you’re done.  
2) **Custom/HPT jobs** — confirm no jobs stuck in `RUNNING`.  
3) **Endpoints** — undeploy models and delete unused endpoints.  
4) **Batch predictions** — ensure no jobs queued or running.  
5) **Artifacts** — delete large intermediate artifacts you won’t reuse.  
6) **GCS** — keep only one “source of truth”; avoid duplicate datasets in multiple buckets/regions.


## Shutting down Vertex AI Workbench notebooks

Vertex AI has two notebook flavors; follow the matching steps:

### Managed Notebooks (recommended for workshops)
- **Console**: Vertex AI → **Workbench** → **Managed notebooks** → select runtime → **Stop**.  
- **Idle shutdown**: Edit runtime → enable **Idle shutdown** (e.g., 60–120 min).  
- **CLI**:
  ```bash
  # List managed runtimes (adjust region)
  gcloud notebooks runtimes list --location=us-central1
  # Stop a runtime
  gcloud notebooks runtimes stop RUNTIME_NAME --location=us-central1
  ```

### User‑Managed Notebooks
- **Console**: Vertex AI → **Workbench** → **User‑managed notebooks** → select instance → **Stop**.  
- **CLI**:
  ```bash
  # List user-managed instances (adjust zone)
  gcloud notebooks instances list --location=us-central1-b
  # Stop an instance
  gcloud notebooks instances stop INSTANCE_NAME --location=us-central1-b
  ```

> **Disks still cost money while the VM is stopped.** Delete old runtimes/instances *and* their disks if you’re done with them.


## Cleaning up training, tuning, and batch jobs

### Audit with CLI
```bash
# Custom training jobs
gcloud ai custom-jobs list --region=us-central1
# Hyperparameter tuning jobs
gcloud ai hp-tuning-jobs list --region=us-central1
# Batch prediction jobs
gcloud ai batch-prediction-jobs list --region=us-central1
```

### Stop/delete as needed
```bash
# Example: cancel a custom job
gcloud ai custom-jobs cancel JOB_ID --region=us-central1
# Delete a completed job you no longer need to retain
gcloud ai custom-jobs delete JOB_ID --region=us-central1
```

> Tip: Keep one “golden” successful job per experiment, then remove the rest to reduce console clutter and artifact storage.

## Undeploy models and delete endpoints (major cost pitfall)

### Find endpoints and deployed models
```bash
gcloud ai endpoints list --region=us-central1
gcloud ai endpoints describe ENDPOINT_ID --region=us-central1
```

### Undeploy and delete
```bash
# Undeploy the model from the endpoint (stops node-hour charges)
gcloud ai endpoints undeploy-model ENDPOINT_ID   --deployed-model-id=DEPLOYED_MODEL_ID   --region=us-central1   --quiet

# Delete the endpoint if you no longer need it
gcloud ai endpoints delete ENDPOINT_ID --region=us-central1 --quiet
```

> **Model Registry**: If you keep models registered but don’t serve them, you won’t pay endpoint node‑hours. Periodically prune stale model versions to reduce storage.


## GCS housekeeping (lifecycle policies, versioning, egress)

### Quick size & contents
```bash
# Human-readable bucket size
gsutil du -sh gs://YOUR_BUCKET
# List recursively
gsutil ls -r gs://YOUR_BUCKET/** | head -n 50
```

### Lifecycle policy example
Keep workshop artifacts tidy by auto‑deleting temporary outputs and capping old versions.

1) Save as `lifecycle.json`:
```json
{
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
```
2) Apply to bucket:
```bash
gsutil lifecycle set lifecycle.json gs://YOUR_BUCKET
gsutil lifecycle get gs://YOUR_BUCKET
```

### Egress reminder
Downloading out of GCP (to local machines) incurs egress charges. Prefer **in‑cloud** training/evaluation and share results via GCS links.



## Labels, budgets, and cost visibility

### Standardize **labels** on all resources
Use the same labels everywhere (notebooks, jobs, buckets) so billing exports can attribute costs.

- Examples: `owner=yourname`, `team=ml-workshop`, `purpose=titanic-demo`, `env=dev`
- CLI examples:
  ```bash
  # Add labels to a custom job on creation (Python SDK supports labels, too)
  # gcloud example when applicable:
  gcloud ai custom-jobs create --labels=owner=yourname,purpose=titanic-demo ...
  ```

### Set **budgets & alerts**
- In **Billing → Budgets & alerts**, create a budget for your project with thresholds (e.g., 50%, 80%, 100%).  
- Add **forecast‑based** alerts to catch trends early (e.g., projected to exceed budget).  
- Send email to multiple maintainers (not just you).

### Enable **billing export** (optional but powerful)
- Export billing to **BigQuery** to slice by service, label, or SKU.  
- Build a simple Data Studio/Looker Studio dashboard for workshop visibility.



## Monitoring and alerts (catch leaks quickly)

- **Cloud Monitoring dashboards**: Track notebook VM uptime, endpoint deployment counts, and job error rates.  
- **Alerting policies**: Trigger notifications when:
  - A **Workbench runtime** has been **running > N hours** outside workshop hours.
  - An **endpoint node count > 0** for > 60 minutes after a workshop ends.
  - **Spend forecast** exceeds budget threshold.

> Keep alerts few and actionable. Route to email or Slack (via webhook) where your team will see them.



## Quotas and guardrails

- **Quotas** (IAM & Admin → Quotas): cap GPU count, custom job limits, and endpoint nodes to protect budgets.  
- **IAM**: least privilege for service accounts used by notebooks and jobs; avoid wide `Editor` grants.  
- **Org policies** (if available): disallow costly regions/accelerators you don’t plan to use.



## Automating the boring parts

### Nightly auto‑stop for idle notebooks
Use **Cloud Scheduler** to run a daily command that stops notebooks after hours.

```bash
# Cloud Scheduler job (runs daily 22:00) to stop a specific managed runtime
gcloud scheduler jobs create http stop-runtime-job   --schedule="0 22 * * *"   --uri="https://notebooks.googleapis.com/v1/projects/PROJECT_ID/locations/us-central1/runtimes/RUNTIME_NAME:stop"   --http-method=POST   --oidc-service-account-email=SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com
```

> Alternative: call `gcloud notebooks runtimes list` in a small Cloud Run job, filter by `last_active_time`, and stop any runtime idle > 2h.

### Weekly endpoint sweep
- List endpoints; undeploy any with zero recent traffic (check logs/metrics), then delete stale endpoints.  
- Scriptable with `gcloud ai endpoints list/describe` in Cloud Run or Cloud Functions on a schedule.



## Common pitfalls and quick fixes

- **Forgotten endpoints** → **Undeploy** models; **delete** endpoints you don’t need.  
- **Notebook left running all weekend** → Enable **Idle shutdown**; schedule nightly stop.  
- **Duplicate datasets** across buckets/regions → consolidate; set **lifecycle** to purge `tmp/`.  
- **Too many parallel HPT trials** → cap `parallel_trial_count` (2–4) and increase `max_trial_count` gradually.  
- **Orphaned artifacts** in Artifact Registry/GCS → prune old images/artifacts after promoting a single “golden” run.



:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1 — Find and stop idle notebooks
List your notebooks and identify any runtime/instance that has likely been idle for >2 hours. Stop it via CLI.

**Hints**: `gcloud notebooks runtimes list`, `gcloud notebooks instances list`, `... stop`

:::::::::::::::: solution

Use `gcloud notebooks runtimes list --location=REGION` (Managed) or `gcloud notebooks instances list --location=ZONE` (User‑Managed) to find candidates, then stop them with the corresponding `... stop` command.

::::::::::::::::

::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2 — Write a lifecycle policy
Create and apply a lifecycle rule that (a) deletes objects under `tmp/` after 7 days, and (b) retains only 3 versions of any object.

**Hint**: `gsutil lifecycle set lifecycle.json gs://YOUR_BUCKET`

:::::::::::::::: solution

Use the JSON policy shown above, then run `gsutil lifecycle set lifecycle.json gs://YOUR_BUCKET` and verify with `gsutil lifecycle get ...`.

:::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge 3 — Endpoint sweep
List deployed endpoints in your region, undeploy any model you don’t need, and delete the endpoint if it’s no longer required.

**Hints**: `gcloud ai endpoints list`, `... describe`, `... undeploy-model`, `... delete`

:::::::::::::::: solution

`gcloud ai endpoints list --region=REGION` → pick `ENDPOINT_ID` → `gcloud ai endpoints undeploy-model ENDPOINT_ID --deployed-model-id=DEPLOYED_MODEL_ID --region=REGION --quiet` → if not needed, `gcloud ai endpoints delete ENDPOINT_ID --region=REGION --quiet`.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Endpoints and running notebooks are the most common cost leaks; undeploy/stop first.
- Prefer **Managed Notebooks** with **Idle shutdown**; schedule nightly auto‑stop.
- Keep storage tidy with **GCS lifecycle policies** and avoid duplicate datasets.
- Standardize **labels**, set **budgets**, and enable **billing export** for visibility.
- Use `gcloud`/`gsutil` to audit and clean quickly; automate with Scheduler + Cloud Run/Functions.

::::::::::::::::::::::::::::::::::::::::::::::::
