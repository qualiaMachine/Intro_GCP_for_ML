---
title: "Hyperparameter Tuning on CHTC"
teaching: 40
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions

- How can I use HTCondor to run many training jobs with different hyperparameters in parallel?
- What are the different ways to parameterize and submit multiple jobs in a single submit file?
- How do I collect and compare results from a hyperparameter sweep?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Write an HTCondor submit file that parameterizes hyperparameters using `$(variable)` syntax.
- Use HTCondor's `queue` command with inline lists, external files, and variable substitution to launch parallel sweeps.
- Collect `metrics.json` files from multiple jobs and identify the best trial.
- Understand the trade-offs between grid/random search on CHTC and managed Bayesian optimization services.

::::::::::::::::::::::::::::::::::::::::::::::::

In the previous episode (Episode 5) you submitted a single PyTorch training job to CHTC and inspected its artifacts. That gave you one model trained with one set of hyperparameters. In practice, choices like learning rate, early-stopping patience, and regularization thresholds can dramatically affect model quality — and the best combination is rarely obvious up front.

In this episode we will use HTCondor's **`queue` command** to systematically search for better settings by launching many training jobs in parallel, each with a different combination of hyperparameters. The `train_nn.py` script from Episode 5 already saves a `metrics.json` file with final validation accuracy and loss — we just need to run it many times and compare the results.

### Why CHTC is great for hyperparameter tuning

Hyperparameter tuning is an *embarrassingly parallel* problem: each trial is completely independent, so you can run them all at the same time. This is exactly the kind of workload CHTC is built for. Key advantages:

- **Massive parallelism** — CHTC can run hundreds of independent jobs simultaneously across its shared pool. A sweep that would take hours sequentially can finish in the time of a single trial.
- **No cost** — all of these jobs are free for UW-Madison researchers. There are no credits to burn, no billing surprises, and no reason to limit your search space to save money.
- **Simple to set up** — HTCondor's `queue` syntax makes it straightforward to parameterize jobs without writing custom orchestration code.

Unlike managed services that use Bayesian optimization to choose the next trial based on previous results, CHTC sweeps are essentially **grid search** or **random search** — every combination is decided up front and launched independently. This sounds less sophisticated, but CHTC's massive parallelism more than compensates: you can afford to explore a much larger space when each trial is free and runs in parallel.

### Key steps for hyperparameter tuning on CHTC

1. Write a submit file that uses `$(variable)` placeholders for hyperparameters.
2. Define the combinations to try (inline, in a file, or programmatically).
3. Submit all trials with a single `condor_submit` command.
4. Collect `metrics.json` from each job's output and find the best trial.

## Writing a parameterized submit file

HTCondor submit files support **variable substitution** using the `$(variable)` syntax. When you combine this with the `queue ... from` syntax, HTCondor creates one job per line of input, substituting the variables into every field of the submit file.

Here is a complete submit file for a hyperparameter sweep:

```
# File: tune_nn.sub
universe = docker
docker_image = pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

executable = run_training.sh
arguments = --train train_data.npz --val val_data.npz --learning_rate $(lr) --patience $(pat) --epochs 500

transfer_input_files = train_nn.py, run_training.sh, train_data.npz, val_data.npz
transfer_output_remaps = "model.pt = results/model_$(Cluster)_$(Process).pt; metrics.json = results/metrics_$(Cluster)_$(Process).json"

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

log    = logs/tune_$(Cluster)_$(Process).log
output = logs/tune_$(Cluster)_$(Process).out
error  = logs/tune_$(Cluster)_$(Process).err

request_cpus = 1
request_memory = 4GB
request_disk = 2GB

queue lr,pat from params.txt
```

Let's break down the key parts:

- **`$(lr)` and `$(pat)`** — these are variable placeholders. HTCondor replaces them with values from `params.txt` for each job.
- **`$(Cluster)` and `$(Process)`** — built-in HTCondor variables. `$(Cluster)` is the job cluster ID (shared by all jobs from one submission), and `$(Process)` is the index within that cluster (0, 1, 2, ...). Together they create unique file names for each trial's outputs.
- **`transfer_output_remaps`** — this is crucial for collecting results. Without it, every job would write to `model.pt` and `metrics.json` in the same directory, overwriting each other. The remap renames each job's outputs to include the cluster and process IDs, placing them in a `results/` directory.
- **`queue lr,pat from params.txt`** — reads variable values from an external file (one combination per line).

::::::::::::::::::::::::::::::::::::::: callout

### Create output directories before submitting

HTCondor will not create directories for you. Before running `condor_submit`, make sure the `results/` and `logs/` directories exist:

```bash
mkdir -p results logs
```

If these directories do not exist, your jobs will fail at output transfer time.

:::::::::::::::::::::::::::::::::::::::::::::::

## Three approaches to defining hyperparameter combinations

HTCondor's `queue` command is flexible. Here are three ways to specify which combinations to try, from simplest to most powerful.

### Approach 1: Queue with inline variable lists

For a small number of combinations, you can list them directly in the submit file:

```
queue lr,pat from (
    0.001, 10
    0.01, 5
    0.0001, 20
    0.005, 15
    0.001, 5
    0.0005, 10
)
```

This submits 6 jobs, one for each line inside the parentheses. Each line provides a value for `lr` and `pat`, separated by a comma.

**When to use this:** quick experiments with a handful of combinations where you want everything in one file.

### Approach 2: Queue from a file

For larger sweeps, store the parameter combinations in a separate file:

```
# File: params.txt
0.001, 10
0.01, 5
0.0001, 20
0.005, 15
0.001, 5
0.0005, 10
0.01, 10
0.0001, 15
0.005, 20
```

Then reference it in the submit file:

```
queue lr,pat from params.txt
```

This is cleaner for larger sweeps and lets you generate `params.txt` programmatically (e.g., with a Python script that creates a grid or random sample).

**When to use this:** any sweep with more than a few combinations, or when you want to generate combinations with a script.

### Generating params.txt programmatically

You can use a simple Python script to generate a grid of hyperparameter combinations:

```python
# File: make_params.py
import itertools

learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
patience_values = [5, 10, 15, 20]

with open("params.txt", "w") as f:
    for lr, pat in itertools.product(learning_rates, patience_values):
        f.write(f"{lr}, {pat}\n")

print(f"Wrote {len(learning_rates) * len(patience_values)} combinations to params.txt")
```

Running `python make_params.py` produces a `params.txt` with 20 combinations (5 learning rates x 4 patience values). You could just as easily sample randomly:

```python
# Random search variant
import random

with open("params.txt", "w") as f:
    for _ in range(20):
        lr = 10 ** random.uniform(-4, -1)   # log-uniform between 0.0001 and 0.1
        pat = random.randint(5, 20)
        f.write(f"{lr:.6f}, {pat}\n")
```

### Approach 3: DAGMan for complex workflows

When your hyperparameter sweep is part of a larger pipeline — for example, you want to preprocess data, run the sweep, and then aggregate results automatically — HTCondor's **DAGMan** (Directed Acyclic Graph Manager) can manage the workflow. DAGMan lets you define dependencies between jobs: job B only starts after job A finishes.

A DAG file for a tune-then-aggregate workflow might look like:

```
# File: tune_pipeline.dag
JOB SWEEP tune_nn.sub
JOB AGGREGATE aggregate_results.sub

PARENT SWEEP CHILD AGGREGATE
```

This ensures all sweep trials complete before the aggregation job runs. We will cover DAGMan in more detail in [Episode 8](08-CLI-workflows.md). For now, the key insight is that DAGMan gives you the ability to chain hyperparameter sweeps with post-processing steps automatically.

## Submitting the sweep

Once your submit file and parameter file are ready, submitting is a single command:

```bash
$ condor_submit tune_nn.sub
Submitting job(s)......
6 job(s) submitted to cluster 12345.
```

HTCondor queues all jobs at once. Depending on pool availability, some or all may start running immediately. You can monitor progress with:

```bash
# Check status of all your jobs
condor_q

# Watch a specific cluster
condor_watch_q 12345

# Check why jobs are idle (waiting for resources)
condor_q -better-analyze 12345
```

::::::::::::::::::::::::::::::::::::::: callout

### How many jobs should I submit?

CHTC can handle hundreds of simultaneous jobs, but be a good citizen of the shared pool:

- **Start small** — submit 5-10 trials first to verify the pipeline works end-to-end (correct outputs, no file transfer errors).
- **Then scale up** — once everything works, submit the full sweep. HTCondor's fair-share scheduling ensures your jobs don't starve other users.
- **Check resource requests** — over-requesting memory or disk means your jobs wait longer to match with available machines. Use `condor_q -better-analyze` to diagnose idle jobs.

:::::::::::::::::::::::::::::::::::::::::::::::

## Collecting and comparing results

After all jobs complete, the `results/` directory will contain pairs of files for each trial:

```bash
$ ls results/
metrics_12345_0.json  model_12345_0.pt
metrics_12345_1.json  model_12345_1.pt
metrics_12345_2.json  model_12345_2.pt
metrics_12345_3.json  model_12345_3.pt
metrics_12345_4.json  model_12345_4.pt
metrics_12345_5.json  model_12345_5.pt
```

Each `metrics_<cluster>_<process>.json` file contains the metrics saved by `train_nn.py`, including `final_val_accuracy`, `final_val_loss`, `learning_rate`, `patience`, and other training details.

### Aggregation script

Here is a simple Python script that reads all metrics files, compares them, and reports the best trial:

```python
# File: find_best_trial.py
import json
import glob
import sys

def find_best_trial(results_dir="results"):
    metrics_files = sorted(glob.glob(f"{results_dir}/metrics_*.json"))

    if not metrics_files:
        print(f"No metrics files found in {results_dir}/")
        sys.exit(1)

    trials = []
    for path in metrics_files:
        with open(path) as f:
            data = json.load(f)
        data["_file"] = path
        trials.append(data)

    # Sort by validation accuracy (descending)
    trials.sort(key=lambda t: t.get("final_val_accuracy", 0), reverse=True)

    print(f"{'File':<40} {'Val Acc':>8} {'Val Loss':>9} {'LR':>10} {'Patience':>9}")
    print("-" * 80)
    for t in trials:
        print(f"{t['_file']:<40} {t.get('final_val_accuracy', 'N/A'):>8.4f} "
              f"{t.get('final_val_loss', 'N/A'):>9.4f} "
              f"{t.get('learning_rate', 'N/A'):>10.6f} "
              f"{t.get('patience', 'N/A'):>9}")

    best = trials[0]
    print(f"\nBest trial: {best['_file']}")
    print(f"  Validation accuracy: {best.get('final_val_accuracy', 'N/A'):.4f}")
    print(f"  Validation loss:     {best.get('final_val_loss', 'N/A'):.4f}")
    print(f"  Learning rate:       {best.get('learning_rate', 'N/A')}")
    print(f"  Patience:            {best.get('patience', 'N/A')}")

    # Identify the corresponding model file
    model_file = best["_file"].replace("metrics_", "model_").replace(".json", ".pt")
    print(f"  Model file:          {model_file}")

if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    find_best_trial(results_dir)
```

Run it after all jobs complete:

```bash
$ python find_best_trial.py results/

File                                     Val Acc  Val Loss         LR  Patience
--------------------------------------------------------------------------------
results/metrics_12345_2.json              0.8212    0.4015   0.000100        20
results/metrics_12345_0.json              0.8101    0.4198   0.001000        10
results/metrics_12345_4.json              0.8045    0.4301   0.001000         5
results/metrics_12345_3.json              0.7989    0.4456   0.005000        15
results/metrics_12345_1.json              0.7877    0.4612   0.010000         5
results/metrics_12345_5.json              0.7821    0.4823   0.000500        10

Best trial: results/metrics_12345_2.json
  Validation accuracy: 0.8212
  Validation loss:     0.4015
  Learning rate:       0.0001
  Patience:            20
  Model file:          results/model_12345_2.pt
```

You can then use the best model file directly for inference or further fine-tuning.

::::::::::::::::::::::::::::::::::::::: callout

### Grid search vs. Bayesian optimization

Managed cloud services like Vertex AI offer **Bayesian optimization**, where the system learns from completed trials to choose more promising hyperparameter combinations for future trials. This is sample-efficient — it finds good results with fewer trials.

On CHTC, we use **grid search** (try every combination in a predefined grid) or **random search** (sample combinations randomly from defined ranges). These methods don't learn from previous results, but they have an important advantage: **every trial is independent**, so they all run in parallel with zero coordination overhead.

When compute is free and plentiful — as it is on CHTC — the practical difference shrinks considerably. You can afford to run 50 or 100 trials instead of 12, covering the search space thoroughly through brute force rather than statistical cleverness.

**Rule of thumb:** Bayesian optimization shines when each trial is expensive (cloud GPU billing by the minute). Grid/random search shines when you have abundant free compute and want simplicity.

:::::::::::::::::::::::::::::::::::::::::::::::

## Putting it all together

Here is the complete workflow, from setup to results:

```bash
# 1. Create directories
mkdir -p results logs

# 2. Generate parameter combinations
python make_params.py

# 3. Verify the params file
cat params.txt

# 4. Submit the sweep
condor_submit tune_nn.sub

# 5. Monitor progress
condor_q

# 6. After all jobs complete, find the best trial
python find_best_trial.py results/
```

::::::::::::::::::::::::::::::::::::: challenge

### Exercise 1: Write a parameter file for a 3-variable sweep

Extend the hyperparameter sweep to include a third variable: `min_delta` (the minimum improvement threshold for early stopping). Write a `params.txt` that includes combinations of:

- `learning_rate`: 0.001, 0.0005, 0.0001
- `patience`: 5, 10, 20
- `min_delta`: 0.001, 0.0001

You will also need to update the submit file to accept the third variable.

1. How many total combinations are there?
2. Write the `params.txt` file (or a script to generate it).
3. Update the `queue` line and `arguments` line of `tune_nn.sub`.

::::::::::::::::::::::: solution

There are 3 x 3 x 2 = **18 combinations**.

A script to generate the file:

```python
import itertools

lrs = [0.001, 0.0005, 0.0001]
pats = [5, 10, 20]
deltas = [0.001, 0.0001]

with open("params.txt", "w") as f:
    for lr, pat, md in itertools.product(lrs, pats, deltas):
        f.write(f"{lr}, {pat}, {md}\n")
```

Updated submit file lines:

```
arguments = --train train_data.npz --val val_data.npz --learning_rate $(lr) --patience $(pat) --min_delta $(md) --epochs 500

queue lr,pat,md from params.txt
```

The `transfer_output_remaps`, log/output/error lines, and everything else stays the same — only `arguments` and `queue` need to change.

:::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Exercise 2: Diagnose a failed sweep

You submitted a sweep of 20 jobs, but when you look at the results directory you only see 15 metrics files. What steps would you take to figure out what happened to the other 5 jobs?

::::::::::::::::::::::: solution

1. **Check job status** — run `condor_q` to see if any jobs are still running, idle, or held:

   ```bash
   condor_q
   ```

2. **Check for held jobs** — held jobs encountered an error. See why:

   ```bash
   condor_q -held
   ```

   Common reasons include: Docker image pull failures, file transfer errors (missing input files), or exceeding requested memory/disk.

3. **Check log files** — each job writes to `logs/tune_<Cluster>_<Process>.log`. The log file records start time, completion, and any abnormal termination. The `.err` file contains stderr output from the job itself:

   ```bash
   # Find which process IDs are missing
   ls results/metrics_*.json | sort

   # Then check the corresponding logs
   cat logs/tune_12345_7.err
   cat logs/tune_12345_7.log
   ```

4. **Resubmit failed jobs** — once you fix the issue, you can resubmit just the failed combinations by creating a new params file with only those lines, or by using `condor_submit` with a specific process range.

:::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Exercise 3: Random search vs. grid search

Instead of a full grid, write a Python script that generates 15 random hyperparameter combinations with:

- `learning_rate`: log-uniform between 0.0001 and 0.01
- `patience`: uniform integer between 5 and 25

Why might random search find a better result than a grid of the same size?

::::::::::::::::::::::: solution

```python
import random

random.seed(42)  # for reproducibility

with open("params_random.txt", "w") as f:
    for _ in range(15):
        lr = 10 ** random.uniform(-4, -2)   # log-uniform: 0.0001 to 0.01
        pat = random.randint(5, 25)
        f.write(f"{lr:.6f}, {pat}\n")

print("Wrote 15 random combinations to params_random.txt")
```

**Why random search can outperform grid search:** Grid search distributes trials evenly across every dimension, which means many trials differ in only one parameter at a time. If one parameter matters much more than the other (e.g., learning rate has a large effect but patience has a small effect), grid search wastes many trials exploring patience values that don't matter, while only testing a few learning rate values.

Random search places trials throughout the full space, so it effectively tests more unique values of each individual parameter. Research by Bergstra and Bengio (2012) showed that random search is more efficient than grid search for the same number of trials when some hyperparameters matter more than others — which is almost always the case in practice.

:::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

## What's next: using your tuned model

After tuning, the best model's weights sit in the `results/` directory. The most common next steps are:

- **Load and evaluate** — load the best `model_*.pt` file in Python and run inference on a test set, just as you did in Episode 5.
- **Move to production** — copy the best model to a shared location or deploy it as part of a larger application.
- **Automate with DAGMan** — set up a DAG that runs the sweep and automatically aggregates results (see [Episode 8](08-CLI-workflows.md)).
- **Iterate** — use the results to narrow your search space and run a more focused sweep around the most promising region.

::::::::::::::::::::::::::::::::::::: keypoints

- HTCondor's `queue ... from` syntax lets you launch many jobs from a single submit file, each with different hyperparameter values substituted via `$(variable)` placeholders.
- Three approaches to defining parameter combinations: inline lists in the submit file, an external parameter file, or DAGMan for multi-step pipelines.
- Use `transfer_output_remaps` with `$(Cluster)` and `$(Process)` to give each trial's output files unique names and avoid overwriting.
- After the sweep completes, a simple Python script can aggregate `metrics.json` files and identify the best trial.
- CHTC's grid/random search trades statistical sophistication for massive free parallelism — run more trials instead of smarter trials.
- All hyperparameter tuning jobs on CHTC are free, removing cost as a constraint on how thoroughly you explore the parameter space.

::::::::::::::::::::::::::::::::::::::::::::::::
