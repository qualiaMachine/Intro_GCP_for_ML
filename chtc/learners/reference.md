---
title: Glossary
---

## Glossary

**Apptainer**
: Container runtime used on CHTC (formerly Singularity). Runs Docker images without root privileges. Used via `container_image` in submit files.

**CHTC**
: Center for High Throughput Computing at UW-Madison. Provides free computing resources to campus researchers.

**Cluster ID**
: A unique numeric identifier assigned by HTCondor when you submit a job. Used to track, monitor, and manage jobs.

**Container image**
: A packaged software environment (OS + libraries + tools) that runs your code reproducibly. Specified as `container_image = docker://image:tag` in submit files.

**`condor_q`**
: HTCondor command to check the status of your submitted jobs. Shows idle, running, and held jobs.

**`condor_submit`**
: HTCondor command to submit a job defined by a submit file (`.sub`).

**DAGMan**
: Directed Acyclic Graph Manager — HTCondor's workflow manager that chains dependent jobs. Submit with `condor_submit_dag`.

**Fair-share priority**
: CHTC's scheduling mechanism that distributes resources equitably. Heavy recent usage temporarily lowers your priority.

**GPU Lab**
: CHTC's pool of GPU-equipped machines, including A100, H100, and H200 GPUs. Access by adding `request_gpus` to your submit file.

**Held job**
: A job that encountered an error and was paused by HTCondor. Check reasons with `condor_q -hold`, fix the issue, and release with `condor_release`.

**HTC (High Throughput Computing)**
: Computing paradigm focused on maximizing total work completed over time through many independent parallel jobs. Contrast with HPC (High Performance Computing).

**HTCondor**
: The job scheduling system used by CHTC to manage and distribute computing work across the pool.

**Interactive job**
: A job that gives you a shell session on a worker machine (`condor_submit -i`). Useful for debugging. Limited to 4 hours on CHTC.

**Process ID**
: Within a cluster of jobs (e.g., from `queue N`), each job gets a process ID starting from 0. Referenced as `$(Process)` in submit files.

**`queue from`**
: HTCondor submit file syntax for submitting multiple jobs with different parameters read from a CSV file. Key feature for hyperparameter sweeps.

**`.sif` file**
: Singularity Image Format — a single file containing an Apptainer container. Built from a `.def` definition file or pulled from Docker Hub.

**Submit file (`.sub`)**
: A text file that tells HTCondor everything about your job: what to run, what files to transfer, and what resources to request.

**Submit node**
: The shared login machine where you write code, prepare data, and submit jobs. Not meant for heavy computation.

**`transfer_input_files`**
: Submit file directive listing files to copy from the submit node to the worker before job execution.

**`transfer_output_files`**
: Submit file directive listing files to copy from the worker back to the submit node after job completion.

**Worker node**
: A machine in the CHTC pool that actually runs your job. You don't log into workers directly — HTCondor assigns jobs to them.
