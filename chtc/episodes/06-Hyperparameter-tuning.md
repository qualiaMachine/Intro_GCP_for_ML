---
title: "Hyperparameter Tuning with HTCondor"
teaching: 25
exercises: 15
---

::::::::::::::::::::::::::::::::::::: questions

- How do I run a hyperparameter sweep on CHTC?
- What is HTCondor's `queue from` syntax and how does it enable parallel sweeps?
- How do I properly evaluate models using train/val/test splits?
- How do I retrain the best model and get a final unbiased performance estimate?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Use HTCondor's `queue from` syntax to launch parallel hyperparameter trials.
- Generate parameter combinations using a script or manual CSV.
- Aggregate trial results and identify the best configuration.
- Retrain the best model on combined train+val data and evaluate on a held-out test set.

::::::::::::::::::::::::::::::::::::::::::::::::

## Why CHTC excels at hyperparameter tuning

Hyperparameter tuning is *embarrassingly parallel* — each trial is completely independent. HTCondor is purpose-built for this pattern: define your parameter grid in a CSV file, and `queue from` launches one job per row. All trials run in parallel, and every trial is free.

| Feature | CHTC HP Tuning |
|---------|----------------|
| **Cost per trial** | **$0** |
| **Max parallel trials** | Hundreds+ (pool capacity) |
| **Search strategy** | Grid or random (you define the CSV) |
| **Configuration** | CSV file + submit file |

Cloud platforms offer managed HP tuning services with Bayesian optimization (smarter trial selection), but they charge per trial and limit parallelism. On CHTC, you can run **far more trials** for free — and more trials often compensates for less-efficient search strategies.

## The proper ML workflow

Before diving into the mechanics, let's establish the correct methodology. A common mistake is tuning hyperparameters on a validation set and then reporting that same validation accuracy as your final result. This is **optimistic** — you've already selected the model that performed best on that data.

The proper workflow uses **three splits**:

```
┌──────────────────────────────────────────────────────────┐
│                    Full Dataset (891 rows)                │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Training    │  │  Validation   │  │     Test     │   │
│  │   60% (534)   │  │  20% (178)    │  │  20% (179)   │   │
│  │              │  │              │  │              │   │
│  │  Fit model   │  │  Early stop  │  │  Final eval  │   │
│  │  weights     │  │  + HP select │  │  (touch once)│   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────────────────────────────────────────┘
```

| Split | Purpose | When used |
|-------|---------|-----------|
| **Train** (60%) | Fit model weights (gradient updates) | Every epoch of every trial |
| **Validation** (20%) | Early stopping + hyperparameter selection | End of each epoch; used to pick best trial |
| **Test** (20%) | Final unbiased performance estimate | Once, at the very end |

**Key rule:** Never use the test set to make decisions. It exists only to give you an honest estimate of how your model will perform on unseen data.

## Step 1: Prepare train/val/test splits

```bash
python3 prepare_data.py --input titanic_train.csv --val_size 0.2 --test_size 0.2
```

This produces three files:
- `train_data.npz` — 60% of the data (for model fitting)
- `val_data.npz` — 20% (for early stopping and HP selection)
- `test_data.npz` — 20% (held out until the final step)

## Step 2: Generate parameter combinations

Create a CSV where each row defines one trial's hyperparameters:

```bash
python3 generate_params.py --output params.csv --mode grid
```

This produces a file like:

```
0.0001, 10, 0.0001
0.0001, 10, 0.001
0.0001, 20, 0.0001
0.0001, 20, 0.001
0.0005, 10, 0.0001
0.0005, 10, 0.001
...
```

Each row: `learning_rate, patience, min_delta`. The grid search produces 24 combinations (4 × 3 × 2). For random search with more trials:

```bash
python3 generate_params.py --output params.csv --mode random --n_trials 50
```

## Step 3: The sweep submit file

This is the key file — it uses HTCondor's `queue from` syntax:

```
# hpt_sweep.sub — Hyperparameter sweep

universe     = vanilla
executable   = run_nn.sh

log          = trial_$(Process)/nn_$(Cluster)_$(Process).log
output       = trial_$(Process)/nn_$(Cluster)_$(Process).out
error        = trial_$(Process)/nn_$(Cluster)_$(Process).err

initialdir   = trial_$(Process)

request_cpus   = 1
request_memory = 2GB
request_disk   = 2GB

transfer_input_files = train_nn.py, run_nn.sh, train_data.npz, val_data.npz
transfer_output_files = model.pt, metrics.json, eval_history.csv

container_image = docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

arguments = --epochs 500 --learning_rate $(learning_rate) --patience $(patience) --min_delta $(min_delta)

queue learning_rate, patience, min_delta from params.csv
```

**How it works:**
1. `queue ... from params.csv` reads each row of the CSV.
2. For each row, it sets the variables `$(learning_rate)`, `$(patience)`, `$(min_delta)`.
3. These variables are substituted into `arguments` and passed to the training script.
4. `$(Process)` is 0, 1, 2, ... for each row, creating separate output directories.
5. All trials are submitted as a single job cluster and run in parallel.

Notice that only `train_data.npz` and `val_data.npz` are transferred — the test set is **not** available to any sweep trial.

## Step 4: Submit the sweep

```bash
# Create output directories for each trial
for i in $(seq 0 $(($(wc -l < params.csv) - 1))); do
    mkdir -p trial_$i
done

# Submit all trials
condor_submit hpt_sweep.sub
```

You'll see something like:

```
Submitting job(s)........................
24 job(s) submitted to cluster 12345678.
```

Monitor the sweep:

```bash
# Watch all trials
condor_watch_q

# See how many are running vs idle
condor_q -batch
```

## Step 5: Aggregate results and select the best

After all trials complete, each `trial_N/` directory contains a `metrics.json` file. The aggregation script collects them and saves the best configuration:

```bash
python3 aggregate_results.py --results_dir . --output_csv hp_summary.csv --output_best best_config.json
```

Output:

```
=== Top 5 Trials (by validation loss) ===
   learning_rate  patience  min_delta  final_val_loss  final_val_accuracy  best_epoch  stopped_epoch
0         0.001        20      0.001        0.421234            0.815642          87             107
1         0.0005       20      0.001        0.425678            0.810056          95             115
...

=== Best Trial ===
  Learning rate: 0.001
  Patience:      20
  Min delta:     0.001
  Val loss:      0.421234
  Val accuracy:  0.815642

Best config saved to: best_config.json
```

The `best_config.json` file captures the winning hyperparameters for the next step.

::::::::::::::::::::::::::::::::::::: callout

### Validation accuracy is not your final result

At this point you might be tempted to report 81.6% accuracy and call it done. But this number is **optimistically biased**: you selected this configuration *because* it had the best validation score out of 24 trials. The more trials you run, the more likely you are to find one that got lucky on the validation set.

The test set gives you an honest, unbiased estimate of real-world performance.

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 6: Retrain on train+val and evaluate on test

Now that you've selected the best hyperparameters, you want to:

1. **Retrain** using the winning configuration on the **combined train+val data** — this gives the model more training examples, which typically improves performance.
2. **Evaluate** on the **held-out test set** — this gives you an unbiased performance estimate.

The `retrain_best.py` script handles both steps:

```bash
# Run locally on the submit node (quick for Titanic-sized data)
python3 retrain_best.py --config best_config.json

# Or submit as an HTCondor job
condor_submit retrain_best.sub
```

What `retrain_best.py` does:
1. Reads `best_config.json` to get the winning hyperparameters.
2. Calls `train_nn.py` with `--combine_train_val` — this concatenates train+val into one training set (giving the model ~80% of the data instead of 60%).
3. Trains for `best_epoch × 1.2` epochs (slightly more than the sweep, since there's more training data).
4. Passes `--test test_data.npz` — after training, the script evaluates on the held-out test set.

Output:

```
=== Retraining with Best Configuration ===
  Learning rate: 0.001
  Patience:      20
  Min delta:     0.001
  Best epoch:    87
  Retrain epochs: 104 (best_epoch=87 × 1.2)

[INFO] Combining train + val data for final retraining
[INFO] Combined training set: 712 rows
Using device: cpu
...
validation_accuracy: 0.820225
validation_loss: 0.418234

test_loss: 0.435123
test_accuracy: 0.804469
```

The **test accuracy** (80.4%) is your final, honest performance number. Notice it's slightly lower than the validation accuracy — this is normal and expected. If the gap were very large, it would suggest overfitting to the validation set.

## The complete workflow summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. prepare_data.py         → train.npz, val.npz, test.npz  │
│ 2. generate_params.py      → params.csv                    │
│ 3. condor_submit sweep.sub → trial_0/ trial_1/ ... trial_N/│
│ 4. aggregate_results.py    → hp_summary.csv, best_config   │
│ 5. retrain_best.py         → final model.pt + test metrics │
└─────────────────────────────────────────────────────────────┘
```

In Episode 8 (CLI Workflows), we'll automate this entire pipeline as a single DAGMan workflow.

## Scaling up

One of CHTC's greatest strengths: scaling from 12 to 120 trials requires **only changing the CSV file**. No quota increases, no budget approvals, no code changes.

```bash
# Generate 100 random trials
python3 generate_params.py --output params.csv --mode random --n_trials 100

# Submit — HTCondor handles parallelism
condor_submit hpt_sweep.sub
```

On commercial cloud platforms, 100 CPU trials might cost $1–$2. On CHTC, it costs $0.

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Run the full workflow

1. Prepare data with train/val/test splits.
2. Generate a grid of 12 parameter combinations.
3. Submit the sweep.
4. Aggregate results and identify the best trial.
5. Retrain the best model on train+val and evaluate on the test set.

:::::::::::::::: solution

```bash
# Step 1: Prepare data
python3 prepare_data.py --input titanic_train.csv

# Step 2: Generate params
python3 generate_params.py --output params.csv --mode grid

# Step 3: Submit sweep
for i in $(seq 0 $(($(wc -l < params.csv) - 1))); do
    mkdir -p trial_$i
done
condor_submit hpt_sweep.sub

# Step 4: After sweep completes, aggregate
python3 aggregate_results.py --results_dir . --output_csv hp_summary.csv --output_best best_config.json

# Step 5: Retrain best model on train+val, evaluate on test
python3 retrain_best.py --config best_config.json
```

Your final `metrics.json` will contain both `test_loss` and `test_accuracy` — the unbiased performance estimate.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Scale to 50 trials

1. Generate 50 random parameter combinations.
2. Submit the sweep.
3. How long did 50 trials take compared to 12? (Check HTCondor logs.)
4. Did the best configuration change compared to the 12-trial grid?

:::::::::::::::: solution

```bash
python3 generate_params.py --output params.csv --mode random --n_trials 50
for i in $(seq 0 49); do mkdir -p trial_$i; done
condor_submit hpt_sweep.sub
```

If the pool has capacity, 50 trials may complete in roughly the same wall-clock time as 12, because they all run in parallel. The total compute time is 50× one trial, but the wall-clock time depends on parallelism and queue wait times.

With 50 random trials, you're more likely to find a configuration that lands in a good region of the hyperparameter space compared to a 12-point grid.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 3: Why not use the test set during the sweep?

A colleague suggests modifying the sweep to evaluate every trial on the test set, so you can pick the trial with the best test accuracy. Why is this a bad idea?

:::::::::::::::: solution

If you select the model with the best **test** accuracy, the test set is no longer independent — it's now a validation set. Your reported "test accuracy" becomes optimistically biased for the same reason the validation accuracy is: you picked the configuration that happened to do best on that specific data.

The whole point of a held-out test set is that it's used **exactly once**, after all decisions have been made. This gives you an honest estimate of how the model will perform on truly unseen data.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 4: Cost comparison

Estimate the cost of running 100 GPU trials on a commercial cloud platform using a T4 GPU at ~$0.35/hr, assuming each trial takes 5 minutes. Compare to CHTC.

:::::::::::::::: solution

- **Commercial cloud:** 100 trials × 5 min × ($0.35/60 min) ≈ **$2.92**
- **CHTC:** 100 trials × $0 = **$0.00**

For a one-time experiment the cloud cost is modest, but for iterative experimentation (running sweeps daily as you develop your model), the savings compound quickly.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- HTCondor's `queue from params.csv` syntax makes hyperparameter sweeps trivial — one job per CSV row, all in parallel.
- Use train/val/test splits: train for fitting, val for early stopping and HP selection, test for final unbiased evaluation.
- After selecting the best HP config, retrain on combined train+val data for maximum training signal.
- The test set should be used exactly once — after all model selection decisions are final.
- CHTC sweeps are free and unlimited — scale from 12 to 1000+ trials without cost or quota concerns.

::::::::::::::::::::::::::::::::::::::::::::::::
