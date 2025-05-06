#!/bin/bash
set -e

# 1) Configure DVC GDrive
/usr/local/bin/configure_dvc_gdrive.sh

# 2) Start MLflow in background
mlflow server --backend-store-uri sqlite:///app/data/mlruns.db \
              --default-artifact-root /app/data/mlruns \
              --host 0.0.0.0 --port 5000 &

# 3) Start Prefect server & agent
prefect server start &
prefect worker start --pool default-pool &

# 4) Finally, run Streamlit
exec streamlit run app/streamlit_app.py \
     --server.port=8501 \
     --server.enableCORS=false \
     --server.enableXsrfProtection=false
