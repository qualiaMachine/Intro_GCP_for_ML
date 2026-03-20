---
title: "Bonus: CLI Workflows Without Notebooks"
teaching: 15
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions

- How do I submit Vertex AI training jobs from the command line instead of a Jupyter notebook?
- What does authentication look like when working outside of a Workbench VM?
- Can I manage GCS buckets, training jobs, and endpoints entirely from a terminal?

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Authenticate with GCP and set a default project using the `gcloud` CLI.
- Upload data to GCS and submit a Vertex AI custom training job from the terminal.
- Monitor, cancel, and clean up jobs using `gcloud ai` commands.
- Understand when CLI workflows are more practical than notebooks.

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: callout

### Bonus episode

This episode is not part of the standard workshop flow. It covers CLI alternatives to the notebook-based workflows from earlier episodes. Contributions and feedback are welcome — open an issue or pull request on the [lesson repository](https://github.com/qualiaMachine/Intro_GCP_for_ML).

::::::::::::::::::::::::::::::::::::::::::::::::::

## Why use the CLI?

Throughout this workshop we used Jupyter notebooks on a Vertex AI Workbench VM as our control center. That setup is great for teaching, but it is not the only way — and sometimes it is not the best way. Common situations where a terminal-based workflow makes more sense:

- **Automation and CI/CD** — You want a GitHub Actions workflow or a cron job to kick off training runs. Notebooks require manual interaction; shell scripts do not.
- **SSH into an HPC cluster or remote server** — You already have a terminal session and do not want to spin up a Workbench VM just to submit a job.
- **Reproducibility** — A shell script checked into version control is easier to review and reproduce than a notebook with hidden state.
- **Cost** — If all you need is to submit a job, paying for a Workbench VM while you wait is unnecessary. You can submit from Cloud Shell (free) or your laptop.

Everything we did with the Python SDK in Episodes 4–6 has an equivalent `gcloud` command. This episode walks through the key ones.


## Step 1: Install and authenticate

If you are on a Workbench VM, the `gcloud` CLI is already installed and authenticated via the VM's service account. On your laptop or another machine you need to install and log in.

### Install the gcloud CLI

Follow the instructions for your platform at [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install). On most systems this is a single installer or package manager command:

```bash
# macOS (Homebrew)
brew install --cask google-cloud-sdk

# Ubuntu / Debian
sudo apt-get install google-cloud-cli

# Windows — download the installer from the link above
```

### Authenticate

```bash
# Interactive browser-based login (laptop / desktop)
gcloud auth login

# Set your default project so you don't need --project on every command
gcloud config set project YOUR_PROJECT_ID

# Set a default region (optional but saves typing)
gcloud config set compute/region us-central1
```

On a Workbench VM these steps are already done for you — the VM's attached service account provides credentials automatically. This is the authentication convenience mentioned in [Episode 2](02-Notebooks-as-controllers.md).

### Application Default Credentials

If you also want to use the Python SDK (e.g., `aiplatform.init()`) outside of a Workbench VM, you need Application Default Credentials (ADC):

```bash
gcloud auth application-default login
```

This stores a credential file locally that Google client libraries pick up automatically. Without it, Python SDK calls from your laptop will fail with an authentication error.


## Step 2: Upload data to GCS

In Episode 3 we uploaded data through the [Cloud Console](https://console.cloud.google.com/storage/browser). From the CLI the equivalent is:

```bash
# Create a bucket (if it doesn't already exist)
gcloud storage buckets create gs://doe-titanic \
    --location=us-central1

# Upload the Titanic CSV files
gcloud storage cp ~/Downloads/data/titanic_train.csv gs://doe-titanic/
gcloud storage cp ~/Downloads/data/titanic_test.csv  gs://doe-titanic/

# Verify
gcloud storage ls gs://doe-titanic/
```

::::::::::::::::::::::::::::::::::::: callout

### gsutil vs gcloud storage

Older tutorials may reference `gsutil`. Google now recommends `gcloud storage` as the primary CLI for Cloud Storage. The commands are very similar (`gsutil cp` → `gcloud storage cp`), but `gcloud storage` is faster for large transfers and receives more active development.

::::::::::::::::::::::::::::::::::::::::::::::::::


## Step 3: Submit a training job

In Episode 4 we used the Python SDK to create and run a `CustomTrainingJob`. The `gcloud` equivalent is `gcloud ai custom-jobs create`. You provide a JSON or YAML config file that describes the job.

### Write a job config file

Create a file called `xgb_job.yaml`:

```yaml
# xgb_job.yaml — Vertex AI custom training job config
# Note: display_name goes on the command line (--display-name), not in this file.
# The --config file describes the job *spec* only, using snake_case field names.
worker_pool_specs:
  - machine_spec:
      machine_type: n1-standard-4
    replica_count: 1
    container_spec:
      image_uri: us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest
      args:
        - "--train=gs://doe-titanic/titanic_train.csv"
        - "--max_depth=6"
        - "--eta=0.3"
        - "--subsample=0.8"
        - "--colsample_bytree=0.8"
        - "--num_round=100"
base_output_directory:
  output_uri_prefix: gs://doe-titanic/artifacts/xgb/cli-run/
```

Replace the bucket name and hyperparameters to match your setup.

### Submit the job

```bash
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=cli-xgb-titanic \
    --config=xgb_job.yaml
```

::::::::::::::::::::::::::::::::::::: callout

### Windows users — line continuation syntax

The `\` at the end of each line is a **Linux / macOS** line continuation character. It does **not** work in the Windows Command Prompt. You have three options:

1. **Put the command on one line** (easiest):

   ```
   gcloud ai custom-jobs create --region=us-central1 --display-name=cli-xgb-titanic --config=xgb_job.yaml
   ```

2. **Use the `^` continuation character** (Windows CMD):

   ```
   gcloud ai custom-jobs create ^
       --region=us-central1 ^
       --display-name=cli-xgb-titanic ^
       --config=xgb_job.yaml
   ```

3. **Use the backtick continuation character** (PowerShell):

   ```
   gcloud ai custom-jobs create `
       --region=us-central1 `
       --display-name=cli-xgb-titanic `
       --config=xgb_job.yaml
   ```

This applies to **all** multi-line commands in this episode, not just this one.

::::::::::::::::::::::::::::::::::::::::::::::::::

Vertex AI provisions a VM, runs your training container, and writes outputs to the `base_output_directory`. The job runs on GCP's infrastructure, not on your machine — you can close your terminal and it keeps going.

### GPU example (PyTorch)

For the PyTorch GPU job from Episode 5, the config includes an `acceleratorType` and `acceleratorCount`. Note that the argument names must match exactly what `train_nn.py` expects (`--train`, `--val`, `--learning_rate`, etc.):

```yaml
# pytorch_gpu_job.yaml
worker_pool_specs:
  - machine_spec:
      machine_type: n1-standard-8
      accelerator_type: NVIDIA_TESLA_T4
      accelerator_count: 1
    replica_count: 1
    container_spec:
      image_uri: us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest
      args:
        - "--train=gs://doe-titanic/data/train_data.npz"
        - "--val=gs://doe-titanic/data/val_data.npz"
        - "--epochs=500"
        - "--learning_rate=0.001"
        - "--patience=50"
base_output_directory:
  output_uri_prefix: gs://doe-titanic/artifacts/pytorch/cli-gpu-run/
```

Submit the same way:

```bash
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=cli-pytorch-titanic-gpu \
    --config=pytorch_gpu_job.yaml
```


## Step 4: Monitor jobs

### List jobs

```bash
gcloud ai custom-jobs list --region=us-central1
```

This prints a table with job ID, display name, state (`JOB_STATE_RUNNING`, `JOB_STATE_SUCCEEDED`, etc.), and creation time.

### Stream logs

```bash
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

This is the CLI equivalent of watching the log panel in a notebook — output streams to your terminal in real time.

### Hyperparameter tuning jobs

The `gcloud ai hp-tuning-jobs` family works the same way:

```bash
gcloud ai hp-tuning-jobs list --region=us-central1
gcloud ai hp-tuning-jobs stream-logs JOB_ID --region=us-central1
```

Creating HP tuning jobs via YAML is more verbose — for complex tuning configs, the Python SDK ([Episode 6](06-Hyperparameter-tuning.md)) is often more readable.


## Step 5: Check for running resources (don't skip this)

The biggest risk with CLI workflows is submitting a job — or leaving a notebook VM running — and forgetting about it. Unlike a Workbench notebook where you can see tabs and running kernels, the CLI gives you no visual reminder that something is still billing you. Jobs and VMs keep running whether or not your terminal is open.

**Get in the habit of checking before you walk away:**

```bash
# Training jobs still running
gcloud ai custom-jobs list --region=us-central1 --filter="state=JOB_STATE_RUNNING"

# HP tuning jobs still running
gcloud ai hp-tuning-jobs list --region=us-central1 --filter="state=JOB_STATE_RUNNING"

# Endpoints still deployed (these bill 24/7, even when idle)
gcloud ai endpoints list --region=us-central1

# Workbench notebook VMs still running
gcloud workbench instances list --location=us-central1-a
```

If anything shows up that you don't need, shut it down:

```bash
# Cancel a running training job
gcloud ai custom-jobs cancel JOB_ID --region=us-central1

# Undeploy a model from an endpoint (stops the per-hour charge)
gcloud ai endpoints undeploy-model ENDPOINT_ID \
    --region=us-central1 \
    --deployed-model-id=DEPLOYED_MODEL_ID

# Stop a Workbench notebook VM
gcloud workbench instances stop INSTANCE_NAME --location=us-central1-a
```

::::::::::::::::::::::::::::::::::::: callout

### Cost leaks are silent

A forgotten endpoint bills ~ `$1.50`–`$3`/hour depending on machine type — that's **`$36`–`$72`/day** doing nothing. A GPU training job you accidentally submitted twice burns money until you cancel it. There's no pop-up warning; you'll only find out on your billing dashboard or when you hit a quota.

Build the habit: **every time you finish a CLI session, run the check commands above.** For a more thorough cleanup checklist, see [Episode 9](09-Resource-management-cleanup.md).

::::::::::::::::::::::::::::::::::::::::::::::::


## Step 6: Download results

After a job succeeds, download artifacts from GCS:

```bash
# List what the job wrote
gcloud storage ls gs://doe-titanic/artifacts/xgb/cli-run/

# Download everything locally
gcloud storage cp -r gs://doe-titanic/artifacts/xgb/cli-run/ ./local_results/
```

You can then load the model and metrics in a local Python session for evaluation — no Workbench VM required.


## Putting it all together: a shell script

Here is a minimal end-to-end script that submits a training job and waits for it to finish. You could check this into your repository or trigger it from CI.

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="your-project-id"
REGION="us-central1"
BUCKET="doe-titanic"
RUN_ID=$(date +%Y%m%d-%H%M%S)

# Upload latest training data
gcloud storage cp ./data/titanic_train.csv gs://${BUCKET}/

# Submit the job
gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name="xgb-${RUN_ID}" \
    --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest \
    --args="--train=gs://${BUCKET}/titanic_train.csv,--max_depth=6,--eta=0.3,--num_round=100" \
    --base-output-dir=gs://${BUCKET}/artifacts/xgb/${RUN_ID}/

echo "Job submitted. Check status with:"
echo "  gcloud ai custom-jobs list --region=${REGION}"
```

::::::::::::::::::::::::::::::::::::: callout

### Cloud Shell — free CLI access

If you do not want to install the `gcloud` CLI locally, you can use **Cloud Shell** directly in the [Google Cloud Console](https://console.cloud.google.com/). It gives you a free, temporary Linux VM with `gcloud` pre-installed and authenticated. Click the terminal icon (">_") in the top-right corner of the Cloud Console to open it.

Cloud Shell is a good option for one-off job submissions or quick resource checks without spinning up a Workbench instance.

::::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1 — Submit a job from the CLI

Using the XGBoost YAML config shown above (adjusted for your bucket name), submit a training job from Cloud Shell or your local terminal. Verify it appears in the Vertex AI Console under **Training > Custom Jobs**.

:::::::::::::::::::::::::::::::::::: solution

```bash
# Edit xgb_job.yaml with your bucket name, then:
gcloud ai custom-jobs create --region=us-central1 --display-name=cli-xgb-titanic --config=xgb_job.yaml

# Confirm it's running:
gcloud ai custom-jobs list --region=us-central1
```

:::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2 — Stream logs in real time

Find the job ID from the previous challenge and stream its logs to your terminal. Compare this experience to watching logs in the notebook.

:::::::::::::::::::::::::::::::::::: solution

```bash
# Get the job ID from the list output
gcloud ai custom-jobs list --region=us-central1

# Stream logs (replace JOB_ID with the actual ID)
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

:::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 3 — Download and inspect artifacts

After your job completes, download the model and metrics files to your local machine. Load `metrics.json` in Python and verify the accuracy value.

:::::::::::::::::::::::::::::::::::: solution

```bash
gcloud storage cp -r gs://YOUR_BUCKET/artifacts/xgb/cli-run/ ./results/
python3 -c "import json; print(json.load(open('./results/model/metrics.json')))"
```

:::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::


## When to use notebooks vs. CLI

| | Notebooks | CLI / scripts |
|---|---|---|
| **Best for** | Exploration, teaching, visualization | Automation, CI/CD, reproducibility |
| **Auth setup** | Automatic on Workbench VMs | Requires `gcloud auth login` or service account keys |
| **Cost** | Pay for VM uptime while notebook is open | Free from Cloud Shell; zero cost from laptop |
| **State management** | Hidden state can cause issues | Stateless scripts are easier to debug |
| **Interactivity** | Rich (plots, widgets, markdown) | Terminal only (or pipe to other tools) |

Most real-world ML/AI projects use both: notebooks for early experimentation and CLI/scripts for production runs.

::::::::::::::::::::::::::::::::::::: keypoints

- Every Vertex AI operation available in the Python SDK has an equivalent `gcloud` CLI command.
- `gcloud ai custom-jobs create` submits training jobs from any terminal — no notebook required.
- Use `gcloud auth login` and `gcloud auth application-default login` to authenticate outside of Workbench VMs.
- Cloud Shell provides free, pre-authenticated CLI access directly in the browser.
- Shell scripts checked into version control are more reproducible than notebooks with hidden state.
- CLI workflows give no visual reminder of running resources — always check for active jobs, endpoints, and VMs before walking away.
- Notebooks and CLI workflows are complementary — use each where it fits best.

::::::::::::::::::::::::::::::::::::::::::::::::::
