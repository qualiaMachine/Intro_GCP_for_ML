---
title: "Using a GitHub Personal Access Token (PAT) to Push/Pull from a Vertex AI Notebook"
teaching: 25
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can I securely push/pull code to and from GitHub within a Vertex AI Workbench notebook?  
- What steps are necessary to set up a GitHub PAT for authentication in GCP?  
- How can I convert notebooks to `.py` files and ignore `.ipynb` files in version control?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Configure Git in a Vertex AI Workbench notebook to use a GitHub Personal Access Token (PAT) for HTTPS-based authentication.  
- Securely handle credentials in a notebook environment using `getpass`.  
- Convert `.ipynb` files to `.py` files for better version control practices in collaborative projects.  

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 0: Initial setup
In the previous episode, we cloned our forked repository as part of the [workshop setup](../setup.html). In this episode, we'll see how to push our code to this fork. Complete these three setup steps before moving forward.

1. Clone the fork if you haven't already. See previous episode.  

2. Start a new Jupyter notebook, and name it something like `Interacting-with-git.ipynb`. We can use the default Python 3 kernel in Vertex AI Workbench.  

3. Change directory to the workspace where your repository is located. In Vertex AI Workbench, notebooks usually live under `/home/jupyter/`.  

```python
%cd /home/jupyter/
```

## Step 1: Using a GitHub personal access token (PAT) to push/pull from a Vertex AI notebook
When working in Vertex AI Workbench notebooks, you may often need to push code updates to GitHub repositories. Since Workbench VMs may be stopped and restarted, configurations like SSH keys may not persist. HTTPS-based authentication with a GitHub Personal Access Token (PAT) is a practical solution. PATs provide flexibility for authentication and enable seamless interaction with both public and private repositories directly from your notebook.  

> **Important Note**: Personal access tokens are powerful credentials. Select the minimum necessary permissions and handle the token carefully.

#### Generate a personal access token (PAT) on GitHub
1. Go to **Settings** in GitHub.  
2. Click **Developer settings** at the bottom of the left sidebar.  
3. Select **Personal access tokens**, then click **Tokens (classic)**.  
4. Click **Generate new token (classic)**.  
5. Give your token a descriptive name and set an expiration date if desired.  
6. **Select minimum permissions**:  
   - Public repos: `public_repo`  
   - Private repos: `repo`  
7. Click **Generate token** and copy it immediately—you won’t be able to see it again.

> **Caution**: Treat your PAT like a password. Don’t share it or expose it in your code. Use a password manager to store it.

#### Use `getpass` to prompt for username and PAT

```python
import getpass

# Prompt for GitHub username and PAT securely
username = input("GitHub Username: ")
token = getpass.getpass("GitHub Personal Access Token (PAT): ")
```

This way credentials aren’t hard-coded into your notebook.

## Step 2: Configure Git settings

```python
!git config --global user.name "Your Name" 
!git config --global user.email your_email@wisc.edu
```

- `user.name`: Will appear in the commit history.  
- `user.email`: Must match your GitHub account so commits are linked to your profile.  

## Step 3: Convert `.ipynb` notebooks to `.py`

Tracking `.py` files instead of `.ipynb` helps with cleaner version control. Notebooks store outputs and metadata, which makes diffs noisy. `.py` files are lighter and easier to review.

1. Install Jupytext.  
```python
!pip install jupytext
```

2. Convert a notebook to `.py`.  
```python
!jupytext --to py Interacting-with-GCS.ipynb
```

3. Convert all notebooks in the current directory.  
```python
import subprocess, os

for nb in [f for f in os.listdir() if f.endswith('.ipynb')]:
    pyfile = nb.replace('.ipynb', '.py')
    subprocess.run(["jupytext", "--to", "py", nb, "--output", pyfile])
    print(f"Converted {nb} to {pyfile}")
```

## Step 4: Add and commit `.py` files

```python
%cd /home/jupyter/your-repo
!git status
!git add .
!git commit -m "Converted notebooks to .py files for version control"
```

## Step 5: Add `.ipynb` to `.gitignore`

```python
!touch .gitignore
with open(".gitignore", "a") as gitignore:
    gitignore.write("\n# Ignore Jupyter notebooks\n*.ipynb\n")
!cat .gitignore
```

Add other temporary files too:  

```python
with open(".gitignore", "a") as gitignore:
    gitignore.write("\n# Ignore cache and temp files\n__pycache__/\n*.tmp\n*.log\n")
```

Commit the `.gitignore`:  

```python
!git add .gitignore
!git commit -m "Add .ipynb and temp files to .gitignore"
```

## Step 6: Syncing with GitHub

First, pull the latest changes:  

```python
!git config pull.rebase false
!git pull origin main
```

If conflicts occur, resolve manually before committing.

Then push with your PAT credentials:  

```python
github_url = f'github.com/{username}/your-repo.git'
!git push https://{username}:{token}@{github_url} main
```

## Step 7: Convert `.py` back to notebooks (optional)

To convert `.py` files back to `.ipynb` after pulling updates:  

```python
!jupytext --to notebook Interacting-with-GCS.py --output Interacting-with-GCS.ipynb
```

:::::::::::::::::::::::::::::::::::::::: challenge

### Challenge: GitHub PAT Workflow

- Why might you prefer using a PAT with HTTPS instead of SSH keys in Vertex AI Workbench?  
- What are the benefits of converting `.ipynb` files to `.py` before committing to a shared repo?  

:::::::::::::::: solution

- PATs with HTTPS are easier to set up in temporary environments where SSH configs don’t persist.  
- Converting notebooks to `.py` results in cleaner diffs, easier code review, and smaller repos without stored outputs/metadata.  

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints 

- Use a GitHub PAT for HTTPS-based authentication in Vertex AI Workbench notebooks.  
- Securely enter sensitive information in notebooks using `getpass`.  
- Converting `.ipynb` files to `.py` files helps with cleaner version control.  
- Adding `.ipynb` files to `.gitignore` keeps your repository organized.  

::::::::::::::::::::::::::::::::::::::::::::::::
