---
title: 'Instructor Notes'
---

## Target Learner Profiles

### Alex — Graduate Researcher in Biology
Alex is a second-year PhD student who trains random forest and XGBoost models on tabular genomics data using scikit-learn on their laptop. Their datasets are growing beyond what fits in RAM, and their advisor has suggested using CHTC. Alex has basic Python skills and has heard of HTCondor but has never used it. They want to learn how to run training jobs on shared hardware without babysitting a terminal session.

### Jordan — Data Scientist at a Research Lab
Jordan has 3 years of experience training deep learning models with PyTorch on a local GPU workstation. They are comfortable with the command line and Git. Their lab wants to scale up hyperparameter tuning using CHTC's GPU Lab. Jordan needs to learn how to submit GPU jobs, run parallel tuning sweeps, and collect results across many HTCondor jobs.

### Sam — Postdoc Exploring LLMs for Literature Review
Sam is a postdoc in environmental science who wants to use retrieval-augmented generation (RAG) to extract information from research papers. They have intermediate Python skills and have used Jupyter notebooks extensively, but have no HTC experience. Sam is primarily interested in the RAG episode but needs the foundational CHTC knowledge from earlier episodes to set up their environment and submit jobs.

## Before the Workshop

### Account setup (1–2 weeks prior)
- Confirm that all participants have CHTC accounts. Submit bulk account requests to CHTC if needed — requests can take several business days to process.
- Alternatively, coordinate with CHTC staff to provide temporary workshop accounts.
- Verify that participants can SSH into a submit node (e.g., `ap2002.chtc.wisc.edu`). Off-campus participants may need the UW VPN.
- Verify GPU Lab access: submit a test GPU job to confirm quota and availability.
- Send a pre-workshop email with setup instructions (SSH access, VPN if needed).

### Test run
- Walk through all episodes end-to-end on a CHTC submit node at least once. CHTC configurations and container availability can change.
- Verify that the Docker images used in episodes are accessible from CHTC (e.g., `python:3.10`, `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`).
- Confirm that `data.zip` and `pdfs_bundle.zip` download correctly from the GitHub repository.
- Test that jobs complete within expected timeframes, including GPU jobs.

## During the Workshop

### Pacing and timing
The lesson is designed for roughly **4–5 hours** of instruction (including short breaks). Suggested time allocation:

| Episode | Teaching + Exercises | Notes |
|---------|---------------------|-------|
| 01 Introduction | 12 min | Keep brief; learners are eager to get hands-on |
| 02 Connecting to CHTC | 30 min | First SSH connection; may need VPN troubleshooting |
| 03 Data Management | 45 min | Filesystem navigation, data staging, first file transfers |
| 04 Training (XGBoost) | 40 min | First HTCondor job; queue wait times vary — fill with discussion |
| 05 Training (PyTorch + GPU) | 30 min | GPU jobs may queue longer; discuss CPU vs GPU during wait |
| 06 Hyperparameter Tuning | 50 min | Many parallel jobs; exercises scale up complexity |
| 07 RAG | 30 min | Can be shortened to a demo if running behind |
| 08 Advanced Workflows (bonus) | 25 min | Optional; skip if short on time |
| 09 Resource Management | 35 min | Important — reinforce good citizenship habits |
| **Total** | **~297 min** | **~5 hours including breaks** |

### Common issues
- **"Can't connect via SSH"**: Most common issue. Check: (1) correct username (NetID), (2) correct submit node hostname, (3) VPN connected if off-campus, (4) CHTC account is active.
- **Jobs stuck in Idle**: This is normal — CHTC is a shared resource. Jobs wait for matching machines. GPU jobs may wait longer during peak times. Use this as a teaching moment about resource sharing.
- **Jobs go to Held state**: Usually a submit file error (wrong Docker image, missing input files, resource request too large). Check with `condor_q -hold` to see the hold reason.
- **Docker image pull failures**: Some images may be large and slow to pull on first use. Use smaller base images when possible (e.g., `python:3.10-slim`).
- **Disk quota exceeded in /home**: Remind learners that /home has a ~20 GB quota. Large outputs should go to /staging or be cleaned up.
- **GPU jobs take longer than expected**: GPU Lab queue times vary. Have a backup plan — the CPU versions of all exercises produce the same results.

### Tips
- Encourage learners to **check job status frequently** with `condor_q` and `condor_watch_q`. This builds good habits.
- Remind learners that the **submit node is shared** — no heavy computation, no large downloads to /home.
- When jobs are queued, use the wait time for discussion questions and challenges.
- For the HP tuning episode, start with a small number of jobs (3–5) to validate the pipeline before scaling up.
- Demonstrate `condor_q -hold` and `condor_rm` early — learners will need these.
- For the RAG episode, if using API-based models, ensure API keys are set up in advance. If using open-source models, ensure the Docker images are tested.

## After the Workshop

- Remind learners to clean up their /home directories (old job outputs, logs, etc.).
- Verify no forgotten jobs are still running: `condor_q -all` filtered by workshop users.
- Collect feedback from learners on pacing and difficulty level.
- Share links to CHTC documentation and support channels for continued learning.
