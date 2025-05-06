#!/usr/bin/env bash
set -e

echo "🔧 Configuring DVC GDrive..."
/usr/local/bin/configure_dvc_gdrive.sh

echo "🧪 Ensuring MLflow storage dirs exist"
mkdir -p /app/data/mlruns
touch    /app/data/mlruns.db
chmod -R a+rw /app/data

echo "⏳ Waiting for inference API at http://localhost:8000/health ...(takes a while)"
until curl --silent --fail http://localhost:8000/health; do
  sleep 1
done
echo "✅ Inference API is ready!"

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
prefect server start --host 0.0.0.0 --port 4200 &

echo "⏳ Waiting for Prefect at http://localhost:4200/health ..."
until curl --silent --fail http://localhost:4200/health; do
  sleep 1
done
echo "✅ Prefect server is healthy!"

echo "🔧 Configuring Prefect server..."
POOL_NAME="default-pool"
QUEUE_NAME="default-pool"

# 1) Create the pool if it doesn’t exist
if ! prefect work-pool ls --output json | grep -q "\"name\":\s*\"$POOL_NAME\""; then
  echo "🔨 Creating Prefect work-pool '$POOL_NAME' (process)…"
  prefect work-pool create "$POOL_NAME" --type process
else
  echo "✅ Prefect work-pool '$POOL_NAME' already exists"
fi

# 2) Create the queue if it doesn’t exist
if ! prefect work-queue ls --output json | grep -q "\"name\":\s*\"$QUEUE_NAME\""; then
  echo "🔨 Creating Prefect work-queue '$QUEUE_NAME' in pool '$POOL_NAME'…"
  prefect work-queue create "$QUEUE_NAME" --pool "$POOL_NAME"
else
  echo "✅ Prefect work-queue '$QUEUE_NAME' already exists"
fi

export PREFECT_API_URL="http://localhost:4200/api"
echo "🚀 Starting Prefect worker on queue '$QUEUE_NAME'…"
prefect worker start --work-queue "$QUEUE_NAME" &

# give the worker a moment to spin up
sleep 2
echo "✅ Prefect worker launched!"

echo "🚀 Launching Streamlit UI..."
exec streamlit run app/streamlit_app.py \
     --server.port=8501 \
     --server.enableCORS=false \
     --server.enableXsrfProtection=false
