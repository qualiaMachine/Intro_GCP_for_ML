---
title: Setup
---

## CHTC Account Setup

Before the workshop, you need an active CHTC account with SSH access. Follow these steps:

### 1. Request a CHTC account

If you don't already have a CHTC account:

1. Visit [CHTC Account Request](https://chtc.cs.wisc.edu/uw-research-computing/form)
2. Fill out the form with your UW-Madison details
3. Account provisioning typically takes 1–2 business days
4. You'll receive an email with your username and login instructions

**Do this at least one week before the workshop.**

### 2. Test SSH access

Once your account is active, verify you can connect:

```bash
ssh <username>@submit1.chtc.wisc.edu
```

You should land in your home directory (`/home/<username>/`). If you have issues, contact [CHTC support](mailto:chtc@cs.wisc.edu).

### 3. Set up SSH keys (recommended)

For passwordless login:

```bash
# On your local machine
ssh-keygen -t ed25519 -C "your_email@wisc.edu"
ssh-copy-id <username>@submit1.chtc.wisc.edu
```

### 4. Clone the workshop repository

After logging in:

```bash
cd /home/$USER/
git clone https://github.com/qualiaMachine/Intro_GCP_for_ML.git
```

The CHTC materials are in the `chtc/` directory.

### 5. Verify your storage

Check that you can access all three storage tiers:

```bash
ls /home/$USER/         # Should exist
ls /scratch/$USER/      # Should exist (create if not: mkdir -p /scratch/$USER)
ls /staging/$USER/      # Should exist (create if not: mkdir -p /staging/$USER)
```

## Software requirements

All workshop software runs inside containers on CHTC workers. You don't need to install anything on the submit node beyond what's already available (Python 3, git, text editors).

For optional local development, the Python packages used in this workshop are:
- `pandas`, `numpy`, `scikit-learn`
- `xgboost`
- `torch` (PyTorch)
- `sentence-transformers` (for RAG episode)
- `transformers` (for RAG episode)

## Terminal recommendations

- **macOS/Linux:** Built-in Terminal with SSH
- **Windows:** Windows Terminal with WSL, or PuTTY
- **All platforms:** VS Code with Remote-SSH extension (connect to submit node, edit files in a GUI)

## What to bring

- A laptop with SSH capability
- Your CHTC account credentials
- Curiosity about high throughput computing!
