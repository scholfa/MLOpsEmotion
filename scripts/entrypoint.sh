#!/usr/bin/env bash
set -e

echo "🔧 Configuring DVC GDrive..."
/usr/local/bin/configure_dvc_gdrive.sh

echo "🧪 Ensuring MLflow storage dirs exist"
mkdir -p /app/data/mlruns
touch    /app/data/mlruns.db
chmod -R a+rw /app/data

echo "⏳ Waiting for inference API at http://inference:8000/health ...(takes a while)"
until curl --silent --fail http://inference:8000/health; do
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

echo "🔨 Creating Prefect work-pool '$POOL_NAME'..."
prefect work-pool create --type process "$POOL_NAME" &
sleep 5

echo "✅ Check work-pool '$POOL_NAME'..."
prefect work-pool ls
sleep 5

echo "🔨 Creating Prefect worker …"
prefect worker start --pool "$POOL_NAME" &
sleep 5
echo "✅ Prefect worker launched!"

echo "🔨 Deploying Prefect flow..."
python flows/dvc_pipeline.py &

echo "✅ Prefect flow deployed!"

echo "🚀 Launching Streamlit UI..."
exec streamlit run app/streamlit_app.py \
     --server.port=8501 \
     --server.enableCORS=false \
     --server.enableXsrfProtection=false
