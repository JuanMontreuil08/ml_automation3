---
title: ML CD Test
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "5.49.1"     # o quítalo si no quieres fijar versión
app_file: app.py     # <-- tu archivo con Gradio (está en la raíz)
pinned: false
---

# SMS Spam Classifier using MLOps

This project implements an SMS classification model (spam vs. non-spam) using **XGBoost** and **MLOps practices** to automate the entire machine learning lifecycle.

**What is MLOps?**

MLOps stands for Machine Learning Operations and refers to a set of best practices to manage the entire ML lifecycle more efficiently, from development to deployment and maintenance. It is designed to ensure that machine learning models are developed, tested, and deployed consistently and reliably.

## Features

- Optimized **XGBoost** model training.
- Text transformation using TF-IDF.
- Automated **training, evaluation, and deployment** pipeline.
- **Continuous Integration (CI)**: automated testing and validation.
- **Continuous Deployment (CD)**: seamless model updates to production.

## GitHub Actions (MLOps)

- **🔍 CI - Model Testing (`ci.yml`)**  
  Runs automatically on every push or pull request to the `main` branch. It sets up a Python 3.11 environment, installs all dependencies, downloads required NLTK resources, and executes unit tests using `pytest` to ensure code quality and reliability.

- **📦 CD - Model Deployment (`deploy.yml`)**  
  Automatically syncs the latest version of the project to the **Hugging Face Space** whenever key files (like `app.py`, `train.py`, or model artifacts) are updated on the `main` branch. It installs `git-lfs` and `huggingface_hub`, securely clones the target Space, and uses `rsync` to push only the necessary updates.

- **📈 Retraining Pipeline (`train.yml`)**  
  Runs automatically every day at 8:00 AM (UTC) or on manual trigger. It installs dependencies, retrains the XGBoost model using `train.py`, and commits the updated model artifact (`models/artifacts.pkl`) back to the repository.

