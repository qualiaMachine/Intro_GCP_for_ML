---
title: Reference
---

## Glossary

This glossary covers the CHTC, HTCondor, and ML terms used in this workshop. Refer back here whenever you hit an unfamiliar term during the episodes.

### High-Throughput Computing Basics

High-Throughput Computing (HTC)
: A computing paradigm focused on running many independent tasks over time, maximizing total throughput rather than speed of a single task. CHTC is designed for this model.

HTCondor
: A workload management system developed at UW-Madison that schedules and manages jobs across a pool of distributed computing resources. It matches job requirements to available machines automatically.

Submit Node (Access Point)
: The server you SSH into to write code, prepare data, and submit jobs. It is a shared resource and should not be used for heavy computation. Also called an "access point" in newer HTCondor documentation.

Execute Node
: A machine in the HTCondor pool where your job actually runs. HTCondor assigns one automatically based on your resource request.

ClassAd
: HTCondor's attribute-value system for describing jobs and machines. Job ClassAds specify requirements (CPUs, memory, GPUs); machine ClassAds advertise available resources. HTCondor matches them to schedule work.

Job Universe
: The execution environment for a job. Common universes include `docker` (runs in a container), `vanilla` (runs directly on the execute node), and `container` (newer container support).

### CHTC Infrastructure

CHTC (Center for High Throughput Computing)
: A research computing center at UW-Madison that provides free, large-scale computing resources to the campus community using HTCondor.

GPU Lab
: CHTC's dedicated pool of GPU-equipped machines, including NVIDIA A100 (40/80 GB), H100 (80 GB), H200 (141 GB), L40, and T4 GPUs. Access requires `+WantGPULab = true` in your submit file.

OSPool (Open Science Pool)
: A national shared computing pool that CHTC users can access for additional capacity via `+WantFlocking = true` or `+WantGlidein = true`.

### CHTC Storage

/home
: Your home directory on the submit node (~20 GB quota). Used for code, submit files, and small input/output files. Files here are available on the submit node but must be explicitly transferred to jobs.

/staging
: A larger storage area for datasets and outputs that are too big for /home. Files are transferred to/from jobs using HTCondor's file transfer mechanism or accessed via staging protocols.

SQUID
: A web proxy cache for distributing large, read-only files to many jobs efficiently. Files placed on SQUID are served via HTTP, avoiding repeated file transfers.

### HTCondor Job Management

Submit File (.sub)
: A configuration file that tells HTCondor what to run, what resources to request, what files to transfer, and where to write output. Submitted with `condor_submit`.

condor_submit
: The command to submit a job (or batch of jobs) described by a submit file.

condor_q
: The command to check the status of your submitted jobs (Idle, Running, Held, Completed).

condor_rm
: The command to remove (cancel) one or more of your jobs from the queue.

condor_status
: The command to view available resources in the HTCondor pool (machines, GPUs, etc.).

condor_history
: The command to view information about your previously completed jobs.

condor_watch_q
: An interactive, auto-refreshing version of `condor_q` that updates in real time.

DAGMan (Directed Acyclic Graph Manager)
: HTCondor's built-in workflow manager for running multi-step jobs with dependencies. Defined in `.dag` files and submitted with `condor_submit_dag`.

### Container and Environment Terms

Docker
: A container platform for packaging code and dependencies together. CHTC's HTCondor supports running jobs inside Docker containers pulled from Docker Hub or other registries.

Apptainer (formerly Singularity)
: An alternative container runtime commonly used in HPC environments. CHTC supports Apptainer containers as well.

Container Image
: A pre-built package containing an operating system, libraries, and tools. Specified in your submit file (e.g., `docker_image = pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`).

### ML/AI Workflow Terms

Training Job
: A compute task that fits a model to data. On CHTC, this is an HTCondor job that runs your training script inside a container on an execute node.

Hyperparameter Tuning
: The process of searching for optimal model configuration by running multiple training jobs with different settings. On CHTC, this leverages HTCondor's `queue` mechanism to submit many jobs in parallel.

Checkpointing
: Saving model state periodically during training so that a job can be resumed if interrupted. Important for long-running jobs on CHTC where runtime limits apply.

### Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG)
: A pattern where an LLM answers questions by first retrieving relevant passages from a corpus, then generating a response grounded in those passages. This reduces hallucination and allows citation of sources.

Chunking
: The process of breaking a large document into smaller, overlapping text segments so that each segment can be independently embedded and retrieved.

Embedding
: A dense numerical vector (array of floats) that represents the semantic meaning of a piece of text. Texts with similar meanings produce vectors that are close together in the embedding space.

Cosine Similarity
: A measure of how similar two embedding vectors are. Ranges from -1 (opposite) to 1 (identical direction). Used to rank which corpus chunks are most relevant to a query.

## Additional Resources

- [Compute for ML](compute-for-ML.html) — guide to choosing hardware and GPUs on CHTC
- [UW-Madison CHTC Resources](uw-madison-chtc-resources.html) — CHTC support, GPU Lab, and campus computing options
- [Using a GitHub PAT](github-pat.html) — pushing/pulling code from CHTC
- [CHTC Guides](https://chtc.cs.wisc.edu/uw-research-computing/guides) — official CHTC documentation
- [HTCondor Manual](https://htcondor.readthedocs.io/en/latest/) — complete HTCondor reference
