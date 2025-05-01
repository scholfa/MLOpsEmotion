# 🛠 MLOpsEmotion Developer Guide

This guide explains how to set up, develop, and contribute to the MLOpsEmotion project using DVC, Streamlit, MLflow, Prefect, and Google Drive for data storage.

---

## 📁 Project Structure

```
MLOpsEmotion/
├── app/                     # Streamlit UI
├── data/
│   ├── raw/                # Uploaded audio
│   ├── processed/          # Preprocessed files
│   └── inference_output/   # Inference results
├── metadata/               # Logs (e.g. inference_log.json)
├── models/                 # Tracked models
├── scripts/                # Preprocessing, inference, evaluation scripts
├── flows/                  # Prefect orchestration
├── .devcontainer/          # Dev environment config
├── .vscode/                # Optional VS Code settings
├── .secret/                # Local-only secrets (NOT committed)
```

---

## ✅ Quick Start

### 1. Clone and Open in VS Code

```bash
git clone https://github.com/your-org/MLOpsEmotion.git
cd MLOpsEmotion
code .
```

> Make sure you have the **Dev Containers** VS Code extension installed.

---

### 2. Run in Dev Container

When prompted, click **“Reopen in Container”**. VS Code will:

- Build the Docker environment
- Mount your secrets (e.g. GDrive key)
- Set up DVC, Prefect, MLflow, Streamlit

---

### 3. Configure Google Drive Remote

Share a GDrive folder with your **service account email**, then set this in `.env`:

```env
GDRIVE_FOLDER_ID=your-folder-id
```

Mount the downloaded key via:

```json
"mounts": [
  "source=${env:HOME}/.secrets/gdrive-sa.json,target=/app/secret/gdrive-key.json,type=bind"
]
```

DVC will configure the remote automatically using:
`.devcontainer/configure_dvc_gdrive.sh`

---

### 4. Run the System

Inside the container:

```bash
# Run pipeline manually
python flows/dvc_pipeline.py

# Run UI
streamlit run app/streamlit_app.py
```

Or use the **postStartCommand** to auto-run the pipeline.

---

## 📦 Core Tools

- **Streamlit** for the web UI
- **DVC** for data and model versioning
- **MLflow** for metrics logging
- **Prefect** for orchestrating the pipeline
- **Docker** to containerize everything
- **GDrive** (via service account) as your DVC remote

---

## ✅ Best Practices

- Don’t commit `.env`, secrets, or DVC credentials
- Use `dvc push` after every data/model change
- Use `prefect` only inside the container to match the environment
- Use GitHub Actions later for automation

---

## 💡 Useful Commands

```bash
dvc repro               # Run pipeline
dvc push                # Push to GDrive remote
streamlit run app/streamlit_app.py
mlflow ui               # Launch tracking UI
```

---
