---
title: "XGBoost CPU Training on CHTC"
teaching: 20
exercises: 15
---

::::::::::::::::::::::::::::::::::::: questions

- How do I run an XGBoost training job on CHTC using containers?
- How do I set up a submit file for a real ML training workflow?
- How do I retrieve model artifacts after training completes?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Configure an HTCondor submit file for XGBoost training with container support.
- Test a training script locally on the submit node before submitting to the pool.
- Submit, monitor, and retrieve model artifacts from a training job.

::::::::::::::::::::::::::::::::::::::::::::::::

## Overview

In this episode, you'll train an XGBoost model on the Titanic dataset by submitting a training job to CHTC's HTCondor pool. The workflow:

1. Write a Python training script that reads a CSV and saves a model
2. Test it quickly on the submit node
3. Write a submit file specifying the container, resources, and files
4. Submit the job, monitor it, and retrieve the trained model

Because HTCondor handles file transfer, your training script uses plain local file I/O — just `pd.read_csv()` and `joblib.dump()`. No special infrastructure code needed.

## The training script

Here's the CHTC version of `train_xgboost.py` — a simplified version with cloud-specific code stripped out:

```python
#!/usr/bin/env python3
"""train_xgboost.py — Train XGBoost on Titanic data."""

import argparse
import os
from pathlib import Path
from time import time

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib


def preprocess_data(df):
    """Preprocess the Titanic dataset for model training."""
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="titanic_train.csv")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--num_round", type=int, default=100)
    args = parser.parse_args()

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train)
    X, y = preprocess_data(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        "objective": "binary:logistic",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "eval_metric": "logloss",
        "seed": 42,
    }

    start = time()
    model = xgb.train(params, dtrain, num_boost_round=args.num_round)
    print(f"Training time: {time() - start:.2f} seconds")

    output_path = os.path.join(output_dir, "xgboost-model")
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")
```

The script is straightforward: plain `pd.read_csv()` and `joblib.dump()`. No cloud SDKs or special infrastructure code — HTCondor transfers files to and from the worker automatically.

## Test locally first

Before submitting to the pool, always test your script on the submit node with a quick sanity check. This catches most bugs (wrong paths, missing packages, argument errors) without waiting for a job to queue:

```bash
# Quick test on the submit node (don't run the full training here — just verify it starts)
cd /home/$USER/workshop/
python3 train_xgboost.py --train titanic_train.csv --num_round 5
```

If this works (prints training time and saves a model file), your script is ready for HTCondor.

::::::::::::::::::::::::::::::::::::: callout

### Don't train on the submit node

The quick test above uses `--num_round 5` to keep it fast. For real training, always submit to HTCondor — the submit node is shared and not meant for heavy computation.

::::::::::::::::::::::::::::::::::::::::::::::::

## Container setup

CHTC workers don't have XGBoost installed by default. We use a Docker container:

**Option 1: Pull from Docker Hub (simplest)**
```
container_image = docker://continuumio/miniconda3:latest
```
This gives you conda/pip to install packages. For a quicker start, the submit file can use a pre-configured image.

**Option 2: Build a custom Apptainer image**

Create an Apptainer definition file (`xgboost.def`):
```
Bootstrap: docker
From: continuumio/miniconda3:latest

%post
    pip install --no-cache-dir xgboost==2.1.0 scikit-learn pandas joblib numpy

%runscript
    exec python3 "$@"
```

Build it on the submit node:
```bash
apptainer build xgboost.sif xgboost.def
```

Then reference the `.sif` file in your submit file:
```
container_image = xgboost.sif
transfer_input_files = xgboost.sif, ...
```

For this workshop, we'll use Docker Hub directly.

## The submit file

```
# train_xgboost.sub — XGBoost CPU training job

universe     = vanilla
executable   = run_xgboost.sh

log          = xgboost_$(Cluster).log
output       = xgboost_$(Cluster).out
error        = xgboost_$(Cluster).err

request_cpus   = 1
request_memory = 2GB
request_disk   = 2GB

transfer_input_files = train_xgboost.py, run_xgboost.sh, titanic_train.csv
transfer_output_files = xgboost-model

container_image = docker://continuumio/miniconda3:latest

arguments = --max_depth 5 --eta 0.1 --num_round 100

queue 1
```

The wrapper script (`run_xgboost.sh`) prints diagnostic info and calls the Python trainer:

```bash
#!/bin/bash
set -e
echo "=== CHTC XGBoost Training ==="
echo "Hostname: $(hostname)"
python3 train_xgboost.py --train titanic_train.csv --output_dir . "$@"
echo "=== Training complete ==="
```

## Submit and monitor

```bash
chmod +x run_xgboost.sh
condor_submit train_xgboost.sub
```

Monitor with:
```bash
condor_q           # Check status
condor_watch_q     # Live updates
```

Once complete, inspect results:
```bash
cat xgboost_*.out   # Training output and accuracy
ls -la xgboost-model # The saved model artifact
```

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Understand the script

Review `train_xgboost.py`. Identify:

1. Where does the training data come from?
2. Where does the trained model get saved?
3. How does HTCondor get the data to the worker and the model back?

:::::::::::::::: solution

1. The data comes from a local CSV file (`--train titanic_train.csv`). HTCondor copies it to the worker via `transfer_input_files`.
2. The model is saved to the current working directory (`.`) as `xgboost-model` using `joblib.dump()`.
3. `transfer_input_files` sends the CSV to the worker before the job starts. `transfer_output_files` brings the model back to the submit node after the job completes. The script itself has no file transfer logic — it just reads and writes local files.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Train with different hyperparameters

1. Submit a training job with `--max_depth 3 --eta 0.3 --num_round 50`.
2. Submit another with `--max_depth 7 --eta 0.05 --num_round 200`.
3. Compare the validation accuracy from each job's output.

Which configuration performed better? (In Episode 6, we'll automate this kind of comparison.)

:::::::::::::::: solution

Edit the `arguments` line in `train_xgboost.sub` for each run, or create separate submit files:

```bash
# Run 1
condor_submit train_xgboost.sub  # with arguments = --max_depth 3 --eta 0.3 --num_round 50

# Run 2
condor_submit train_xgboost.sub  # with arguments = --max_depth 7 --eta 0.05 --num_round 200
```

Check validation accuracy in each `.out` file. The best configuration depends on the data, but typically moderate depth (5–7) and moderate learning rate (0.05–0.1) work well for Titanic.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Training scripts use plain local file I/O — HTCondor handles file transfer automatically.
- Always test your script locally on the submit node (with minimal settings) before submitting to HTCondor.
- Containers provide reproducible environments: use Docker Hub images or build custom Apptainer `.sif` files.
- The submit file specifies resources, files, container, and arguments — then `condor_submit` launches the job.

::::::::::::::::::::::::::::::::::::::::::::::::
