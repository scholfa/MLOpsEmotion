#!/usr/bin/env bash
set -e

echo "🔧 Configuring DVC GDrive..."
/usr/local/bin/configure_dvc_gdrive.sh

echo "🧪 Ensuring MLflow storage dirs exist"
mkdir -p /app/data/mlruns
touch    /app/data/mlruns.db
chmod -R a+rw /app/data

echo "🚀 Starting MLflow server..."
mlflow server \
  --backend-store-uri sqlite:////app/data/mlruns.db \
  --default-artifact-root /app/data/mlruns \
  --host 0.0.0.0 \
  --port 5000 &

echo "⏳ Waiting for MLflow at http://localhost:5000 ..."
until curl --silent --fail http://localhost:5000; do
  sleep 1
done
echo "✅ MLflow is up!"

echo "🚀 Starting Prefect server..."
prefect server start &

echo "⏳ Waiting for Prefect at http://localhost:4200/health ..."
until curl --silent --fail http://localhost:4200/health; do
  sleep 1
done
echo "✅ Prefect server is healthy!"

export PREFECT_API_URL="http://localhost:4200/api"
echo "🚀 Starting Prefect worker..."
prefect worker start --work-queue default &

# give the worker a moment to spin up
sleep 2
echo "✅ Prefect worker launched!"

echo "🚀 Launching Streamlit UI..."
exec streamlit run app/streamlit_app.py \
     --server.port=8501 \
     --server.enableCORS=false \
     --server.enableXsrfProtection=false
