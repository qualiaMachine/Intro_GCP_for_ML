---
title: Setup
---

## Setup (Complete Before the Workshop)

Before attending this workshop, you'll need to complete a few setup steps to ensure you can follow along smoothly. The main requirements are:

1. **CHTC Account** — Request a CHTC account if you don't already have one.
2. **SSH Access** — Ensure you can SSH into a CHTC submit node.
3. **Titanic Dataset** — The required CSV files are included in the workshop repository.
4. **(Optional) GitHub Account** — Only needed if you want to push your work back to a fork. See the [GitHub PAT guide](github-pat.html) for details.

Details on each step are outlined below.

### 1. CHTC Account

You need an active CHTC account to participate in this workshop. There are two scenarios:

#### Option A) You already have a CHTC account

If you've used CHTC before, you're all set. Verify you can log in by running:

```bash
ssh YOUR_NETID@ap2002.chtc.wisc.edu
```

If you can't connect, contact [chtc@cs.wisc.edu](mailto:chtc@cs.wisc.edu) for help.

#### Option B) You need a new account

1. Visit the [CHTC account request page](https://chtc.cs.wisc.edu/uw-research-computing/form).
2. Fill out the request form. Mention that you're attending an ML workshop if asked about your use case.
3. **Submit your request at least 1 week before the workshop** — account creation requires manual review by CHTC staff.
4. Once approved, you'll receive an email with login instructions.

::::::::::::::::::::::::::::::::::::: callout

### Workshop-specific accounts

When this workshop is taught at UW-Madison (e.g., Machine Learning Marathon, Research Bazaar), the instructors may provide temporary shared accounts or coordinate bulk account creation with CHTC. Wait for a pre-workshop email from the instructor to confirm the setup process.

::::::::::::::::::::::::::::::::::::::::::::::::

### 2. SSH Access

You'll need an SSH client to connect to CHTC:

- **macOS/Linux**: Use the built-in Terminal app. SSH is pre-installed.
- **Windows**: Use [Windows Terminal](https://aka.ms/terminal) (Windows 10+, SSH built-in), [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/), or [MobaXterm](https://mobaxterm.mobatek.net/).

Test your connection before the workshop:

```bash
ssh YOUR_NETID@ap2002.chtc.wisc.edu
```

You'll need to authenticate with your UW-Madison NetID and password. If you're off-campus, you may need to use the [UW-Madison VPN](https://it.wisc.edu/services/wiscvpn/) first.

### 3. Workshop Data

The Titanic dataset and other workshop files are included in the lesson repository. During the workshop, you'll clone the repo directly on the CHTC submit node:

```bash
git clone https://github.com/qualiaMachine/Intro_GCP_for_ML.git
```

The repository contains:
- `data/data.zip` — Titanic dataset (titanic_train.csv, titanic_test.csv)
- `data/pdfs_bundle.zip` — Research papers for the RAG episode
- `scripts/` — Training scripts (train_xgboost.py, train_nn.py)
- `submit_files/` — HTCondor submit file examples

### 4. (Optional) Familiarize Yourself with CHTC

If you want a broader introduction to CHTC before the workshop, explore:

- [CHTC Getting Started Guide](https://chtc.cs.wisc.edu/uw-research-computing/guides)
- [HTCondor Quick Start Tutorial](https://htcondor.readthedocs.io/en/latest/getting-htcondor/using-htcondor-first-time.html)
- [CHTC Hello World Example](https://chtc.cs.wisc.edu/uw-research-computing/helloworld)

This is optional but recommended for those who want to get familiar with the command line and job submission before the workshop.
