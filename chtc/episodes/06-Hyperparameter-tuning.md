---
title: "Hyperparameter Tuning with HTCondor"
teaching: 20
exercises: 15
---

::::::::::::::::::::::::::::::::::::: questions

- How do I run a hyperparameter sweep on CHTC?
- How does HTCondor's `queue from` syntax compare to Vertex AI's HyperparameterTuningJob?
- How do I collect and analyze results from parallel trials?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Use HTCondor's `queue from` syntax to launch parallel hyperparameter trials.
- Generate parameter combinations using a script or manual CSV.
- Aggregate trial results and identify the best configuration.

::::::::::::::::::::::::::::::::::::::::::::::::

## Why CHTC excels at hyperparameter tuning

This is where CHTC genuinely shines compared to cloud platforms. Hyperparameter tuning is *embarrassingly parallel* — each trial is completely independent. HTCondor is purpose-built for this pattern.

| Feature | Vertex AI HP Tuning | CHTC HP Tuning |
|---------|-------------------|----------------|
| **Cost per trial** | ~$0.02 (CPU) to $0.35+ (GPU) | **$0** |
| **Max parallel trials** | Limited by quota | Hundreds+ (pool capacity) |
| **Search strategy** | Bayesian, grid, random | Grid, random (you write the CSV) |
| **Configuration** | Python SDK + metric spec | CSV file + submit file |
| **12 trials on CPU** | ~$0.19 | **$0** |
| **100 trials on CPU** | ~$1.60 | **$0** |

The tradeoff: Vertex AI offers built-in Bayesian optimization (smarter trial selection). CHTC uses brute-force grid or random search. But with free compute, you can run **far more trials** than you could afford on cloud — and more trials often compensates for less-efficient search.

## Step 1: Generate parameter combinations

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

## Step 2: The sweep submit file

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

## Step 3: Submit the sweep

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

## Step 4: Aggregate results

After all trials complete, each `trial_N/` directory contains a `metrics.json` file. The aggregation script collects them:

```bash
python3 aggregate_results.py --results_dir . --output_csv hp_summary.csv
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
```

## Scaling up

One of CHTC's greatest strengths: scaling from 12 to 120 trials requires **only changing the CSV file**. No quota increases, no budget approvals, no code changes.

```bash
# Generate 100 random trials
python3 generate_params.py --output params.csv --mode random --n_trials 100

# Submit — HTCondor handles parallelism
condor_submit hpt_sweep.sub
```

On Vertex AI, 100 CPU trials would cost ~$1.60. On CHTC, it costs $0.

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Run a sweep

1. Generate a grid of 12 parameter combinations using `generate_params.py`.
2. Prepare the `.npz` data files (if not done already).
3. Submit the sweep with `condor_submit hpt_sweep.sub`.
4. Watch the jobs with `condor_watch_q`.
5. After completion, run `aggregate_results.py` and identify the best trial.

:::::::::::::::: solution

```bash
# Generate params
python3 generate_params.py --output params.csv --mode grid

# Create trial directories
for i in $(seq 0 $(($(wc -l < params.csv) - 1))); do
    mkdir -p trial_$i
done

# Submit
condor_submit hpt_sweep.sub

# Wait for completion, then aggregate
python3 aggregate_results.py --results_dir . --output_csv hp_summary.csv
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Scale to 50 trials

1. Generate 50 random parameter combinations.
2. Submit the sweep.
3. How long did 50 trials take compared to 12? (Check HTCondor logs.)

:::::::::::::::: solution

```bash
python3 generate_params.py --output params.csv --mode random --n_trials 50
for i in $(seq 0 49); do mkdir -p trial_$i; done
condor_submit hpt_sweep.sub
```

If the pool has capacity, 50 trials may complete in roughly the same wall-clock time as 12, because they all run in parallel. The total compute time is 50× one trial, but the wall-clock time depends on parallelism and queue wait times.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 3: Cost comparison

Estimate the cost of running 100 GPU trials on Vertex AI using `n1-standard-4` + T4 GPU at ~$0.35/hr, assuming each trial takes 5 minutes. Compare to CHTC.

:::::::::::::::: solution

- **Vertex AI:** 100 trials × 5 min × ($0.35/60 min) ≈ **$2.92**
- **CHTC:** 100 trials × $0 = **$0.00**

For a one-time experiment the cloud cost is modest, but for iterative experimentation (running sweeps daily as you develop your model), the savings compound quickly.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- HTCondor's `queue from params.csv` syntax makes hyperparameter sweeps trivial — one job per CSV row, all in parallel.
- Each trial writes `metrics.json`; an aggregation script collects results and finds the best configuration.
- CHTC sweeps are free and unlimited — scale from 12 to 1000+ trials without cost or quota concerns.
- The tradeoff vs. Vertex AI: no built-in Bayesian optimization, but unlimited free trials often compensates.

::::::::::::::::::::::::::::::::::::::::::::::::
