---
title: Setup
---

## Setup (Complete Before the Workshop)

Before attending this workshop, you'll need to complete a few setup steps to ensure you can follow along smoothly. The main requirements are:

1. **GitHub Account** – Create an account and be ready to fork a repository.  
2. **GCP Access** – Use a **shared Google Cloud project** (if attending the Machine Learning Marathon or Research Bazaar) or sign up for a personal GCP Free Tier account.  
3. **Titanic Dataset** – Download the required CSV files in advance.  
6. **(Optional) Google Cloud Skills Boost** — For a broader overview of GCP, visit the [Getting Started with Google Cloud Fundamentals](https://www.cloudskillsboost.google/course_templates/62) course.

Details on each step are outlined below.

### 1. GitHub Account

You will need a GitHub account to access the code provided during this lesson. If you don't already have a GitHub account, please [sign up for GitHub](https://github.com/) to create a free account.  
Don't worry if you're a little rusty on using GitHub or git; we will only use a couple of git commands during the lesson, and the instructor will guide you through them.

### 2. GCP Access

There are two ways to get access to GCP for this lesson. Please wait for a pre-workshop email from the instructor to confirm which option to choose.

#### Option 1) Shared Google Cloud Project

If you are attending this lesson as part of the **Machine Learning Marathon** or **Research Bazaar**, the instructors will provide access to a shared GCP project for all attendees. You do not need to set up your own account.  

What to expect:
* Before the workshop, you will receive an email invitation to join the shared GCP project.  
* During the lesson, you will log in with your Google account credentials (NetID or Gmail).  
* This setup ensures that all participants have a consistent environment and avoids unexpected billing for attendees.  
* Please use shared credits responsibly — they are limited and reused for future training events.  
  * Stay within the provided exercises and avoid launching additional compute-heavy workloads (e.g., training large language models).  
  * Do not enable additional APIs or services unless instructed.

#### Option 2) GCP Free Tier — Skip If Using Shared Project

If you are attending this lesson as part of the Machine Learning Marathon or Research Bazaar, you can skip this step. Otherwise, please follow these instructions:

1. Go to the [GCP Free Tier page](https://cloud.google.com/free) and click **Get started for free**.  
2. Complete the signup process. The Free Tier includes a $300 credit valid for 90 days and ongoing free usage for some smaller services.  
3. Once your account is ready, log in to the [Google Cloud Console](https://console.cloud.google.com/).  
4. During the lesson, we will enable only a few APIs (Compute Engine, Cloud Storage, and Notebooks).  

Following the lesson should cost well under $15 total if you are using your own credits.

### 3. Download the Data

For this workshop, you will need the **Titanic dataset**, which can be used to train a classifier predicting survival.

1. Please download the following zip file (Right-click → Save as):  
   [data.zip](https://raw.githubusercontent.com/qualiaMachine/Intro_GCP_for_ML/main/data/data.zip)  

2. Extract the zip folder contents (Right-click → Extract all on Windows; double-click on macOS).  

3. Save the two data files (train and test) somewhere easy to access, for example:  
   - `~/Downloads/data/titanic_train.csv`  
   - `~/Downloads/data/titanic_test.csv`  

In the first episode, you will create a Cloud Storage bucket and upload this data to use with your notebook.

### 4. (Optional) Google Cloud Skills Boost — Getting Started with Google Cloud Fundamentals

If you want a broader introduction to GCP before the workshop, consider exploring the [Getting Started with Google Cloud](https://www.cloudskillsboost.google/paths/8) self-paced learning path. It covers the basics of the Google Cloud environment, including project structure, billing, IAM (Identity and Access Management), and common services like Compute Engine, Cloud Storage, and BigQuery. This step is optional but recommended for those that want a broader overview of GCP before diving into ML/AI use-cases.
