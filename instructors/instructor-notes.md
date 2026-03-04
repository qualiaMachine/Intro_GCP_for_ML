---
title: 'Instructor Notes'
---

## Target Learner Profiles

### Alex — Graduate Researcher in Biology
Alex is a second-year PhD student who trains random forest and XGBoost models on tabular genomics data using scikit-learn on their laptop. Their datasets are growing beyond what fits in RAM, and their advisor has suggested moving to cloud compute. Alex has basic Python skills and has heard of GCP but has never used it. They want to learn how to store data in the cloud, run training jobs without babysitting a notebook, and keep costs under control.

### Jordan — Data Scientist at a Research Lab
Jordan has 3 years of experience training deep learning models with PyTorch on a local GPU workstation. They are comfortable with the command line and Git. Their lab has GCP credits and wants to scale up hyperparameter tuning for a new project. Jordan needs to learn how to submit managed training jobs, attach GPUs, and compare tuning trial results without managing infrastructure manually.

### Sam — Postdoc Exploring LLMs for Literature Review
Sam is a postdoc in environmental science who wants to use retrieval-augmented generation (RAG) to extract information from research papers. They have intermediate Python skills and have used Jupyter notebooks extensively, but have no cloud experience. Sam is primarily interested in the RAG episode but needs the foundational GCP knowledge from earlier episodes to set up their environment and manage costs responsibly.

## Before the Workshop

### Account setup (1–2 weeks prior)
- Confirm whether you are using a **shared GCP project** or asking learners to use **Free Tier** accounts.
- If using a shared project (the standard approach at UW-Madison via RCI and ML+X), the recommended onboarding procedure is:
  1. **Create or reuse a Google Group** (e.g., `mlm-workshop-2025@googlegroups.com`) that has the necessary IAM roles on the shared project (at minimum: `Vertex AI User`, `Storage Object Admin`, `Compute Viewer`).
  2. **Add learner Google accounts to the group** — either by collecting emails in advance via a registration form or by adding them during a pre-workshop session.
  3. **Allow time for IAM propagation.** After adding a member to a Google Group, it can take **5–15 minutes** (occasionally up to an hour) for GCP to recognize the new membership and grant access. Plan accordingly:
     - **Ideal:** Add all learners **the day before** the workshop so access is ready by start time.
     - **If adding day-of:** Do it at least 15–30 minutes before the first hands-on episode (episode 02). Use the introduction episode to fill time while permissions propagate.
  4. **Verify access** by having at least one test account confirm they can see the shared project in the Cloud Console before the workshop begins.
  5. **After the workshop**, you can remove learners from the group (or delete it) to revoke access cleanly without touching individual IAM bindings.
- Verify GPU quota in the workshop region (`us-central1`). Request increases for `NVIDIA_TESLA_T4` if needed — quota requests can take 1–3 business days.
- Send a pre-workshop email with setup instructions (GitHub account, GCP access, data download).

### Test run
- Walk through all episodes end-to-end on the shared project at least once. GCP UI changes frequently — confirm that screenshots and console paths still match.
- Verify that the Vertex AI prebuilt container URIs in Episodes 04, 05, and 06 are still available (container tags get deprecated).
- Confirm that `data.zip` and `pdfs_bundle.zip` download correctly from the GitHub repository.

## During the Workshop

### Pacing and timing
The lesson is designed for roughly **5 hours** of instruction (including short breaks). Suggested time allocation:

| Episode | Teaching + Exercises | Notes |
|---------|---------------------|-------|
| 01 Introduction | 12 min | Keep brief; learners are eager to get hands-on |
| 02 Notebooks as Controllers | 30 min | First console interaction; VM creation takes 3–5 min — fill with discussion |
| 03 Data Storage & Access | 50 min | Merged bucket creation + notebook data access; first notebook coding |
| 04 Training (XGBoost) | 40 min | Vertex AI job takes 2–5 min; use wait time for Q&A |
| 05 Training (PyTorch + GPU) | 30 min | GPU jobs may take longer; discuss CPU vs GPU during wait |
| 06 Hyperparameter Tuning | 50 min | Start with 1 trial; exercises have learners scale up |
| 07 RAG | 30 min | Can be shortened to a demo if running behind |
| 08 CLI Workflows (bonus) | 15 min | Optional; skip if short on time |
| 09 Resource Cleanup | 40 min | Critical — do not skip. Learners must clean up resources |
| **Total** | **~297 min** | **~5 hours including breaks** |

### Common issues
- **"I can't see the project"**: If a learner was just added to the Google Group, IAM propagation may still be in progress. Have them wait 5–15 minutes, try an incognito/private browser window, and confirm they are logged into the correct Google account (not a personal Gmail if the group expects a university account).
- **Bucket permission errors**: The most common blocker. Have the `gcloud storage buckets add-iam-policy-binding` commands ready to paste into Cloud Shell.
- **VM creation stuck**: If a Workbench Instance gets stuck in "Provisioning" for >5 min, try a different zone in the same region.
- **GPU quota exceeded**: If T4 quota is unavailable, fall back to CPU-only training for Episodes 05–06. The lesson works without GPUs — it just takes longer.
- **Numpy version mismatch**: The PyTorch kernel sometimes has a numpy 2.x conflict. The fix (`pip install --upgrade --force-reinstall "numpy<2"`) is included in Episode 05.
- **Idle shutdown confusion**: Some learners may find their VM stopped mid-workshop. Remind them to increase idle timeout or restart the instance.

### Tips
- Encourage learners to **add labels/tags to every resource** from the start. This is easy to skip but essential for cost tracking in shared accounts.
- Remind learners that **Vertex AI training jobs take 2–5 minutes just for provisioning** before training begins. This is normal, not an error.
- When walking through Episode 09 (cleanup), verify as a group that no endpoints are left deployed and no VMs are still running. This is now the final episode, so a full teardown is appropriate.
- Episode 09 includes a "Check your spend" section — use this as a live demo so learners can see the Billing Reports dashboard. Walk through the budget alert setup (Challenge 1) together as a class if time permits.
- For the RAG episode, have a backup plan in case the Gemini API is temporarily rate-limited. You can demo from pre-computed outputs.

## After the Workshop

- Verify all shared project resources are cleaned up (notebooks stopped, endpoints deleted, buckets with only intentional data remaining).
- Review billing dashboard for any unexpected charges.
- Collect feedback from learners on pacing and difficulty level.
