---
title: 'Instructor Notes'
---

## Before the Workshop

### Account setup (1–2 weeks prior)
- Confirm whether you are using a **shared GCP project** or asking learners to use **Free Tier** accounts.
- If using a shared project, add all learner Google accounts to the project with appropriate IAM roles (at minimum: `Vertex AI User`, `Storage Object Admin`, `Compute Viewer`).
- Verify GPU quota in the workshop region (`us-central1`). Request increases for `NVIDIA_TESLA_T4` if needed — quota requests can take 1–3 business days.
- Send a pre-workshop email with setup instructions (GitHub account, GCP access, data download).

### Test run
- Walk through all episodes end-to-end on the shared project at least once. GCP UI changes frequently — confirm that screenshots and console paths still match.
- Verify that the Vertex AI prebuilt container URIs in episodes 06, 07, and 08 are still available (container tags get deprecated).
- Confirm that `data.zip` and `pdfs_bundle.zip` download correctly from the GitHub repository.

## During the Workshop

### Pacing and timing
The lesson is designed for roughly **4–5 hours** of instruction. Suggested time allocation:

| Episode | Teaching + Exercises | Notes |
|---------|---------------------|-------|
| 01 Introduction | 15 min | Keep brief; learners are eager to get hands-on |
| 02 Data Storage | 20 min | First console interaction — go slowly |
| 03 Notebooks as Controllers | 30 min | Expect VM creation to take 3–5 min; fill with discussion |
| 04 Accessing Data | 30 min | First notebook coding — check everyone can read from GCS |
| 05 GitHub PAT | 25 min | Can be shortened or made optional if time is tight |
| 06 Training (XGBoost) | 35 min | Vertex AI job takes 2–5 min; use wait time for Q&A |
| 07 Training (PyTorch + GPU) | 30 min | GPU jobs may take longer; discuss CPU vs GPU during wait |
| 08 Hyperparameter Tuning | 45 min | Start with 1 trial; scale to more only if time allows |
| 09 Resource Cleanup | 30 min | Critical — do not skip. Learners must clean up resources |
| 10 RAG | 30 min | Can be shortened to a demo if running behind |

### Common issues
- **Bucket permission errors**: The most common blocker. Have the `gcloud storage buckets add-iam-policy-binding` commands ready to paste into Cloud Shell.
- **VM creation stuck**: If a Workbench Instance gets stuck in "Provisioning" for >5 min, try a different zone in the same region.
- **GPU quota exceeded**: If T4 quota is unavailable, fall back to CPU-only training for episodes 07–08. The lesson works without GPUs — it just takes longer.
- **Numpy version mismatch**: The PyTorch kernel sometimes has a numpy 2.x conflict. The fix (`pip install --upgrade --force-reinstall "numpy<2"`) is included in episode 07.
- **Idle shutdown confusion**: Some learners may find their VM stopped mid-workshop. Remind them to increase idle timeout or restart the instance.

### Tips
- Encourage learners to **add labels/tags to every resource** from the start. This is easy to skip but essential for cost tracking in shared accounts.
- Remind learners that **Vertex AI training jobs take 2–5 minutes just for provisioning** before training begins. This is normal, not an error.
- When walking through episode 09 (cleanup), verify as a group that no endpoints are left deployed and no VMs are still running.
- For the RAG episode, have a backup plan in case the Gemini API is temporarily rate-limited. You can demo from pre-computed outputs.

## After the Workshop

- Verify all shared project resources are cleaned up (notebooks stopped, endpoints deleted, buckets with only intentional data remaining).
- Review billing dashboard for any unexpected charges.
- Collect feedback from learners on pacing and difficulty level.
