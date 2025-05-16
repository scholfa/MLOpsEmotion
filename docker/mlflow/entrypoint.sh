#!/bin/bash
#
#mlflow ui --host 0.0.0.0 &
#
## Wait for MLflow to be up
#until curl -s http://localhost:5000; do
#  echo "Waiting for MLflow UI to be available..."
#  sleep 1
#done
#
## Run initialization and serve the model
#python mlflow_init.py
#
#mlflow models serve -m "models:/EmotionRecognizerHF/Staging" -h 0.0.0.0 -p 5001 --no-conda -timeout 600

echo "Starting MLflow UI and model server with gunicorn..."

# Start MLflow UI with increased timeout
#gunicorn --timeout 3600 --workers 2 --bind 0.0.0.0:5000 mlflow.server:app
#mlflow ui --host 0.0.0.0 &

set -e

# Start MLflow Tracking Server in the background
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root /mlruns &

# Wait for MLflow UI to be up
until curl -s http://localhost:5000; do
  echo "Waiting for MLflow UI to be available..."
  sleep 1
done


echo "Running initialization script..."
# Run your custom initialization
python mlflow_init.py

# Start MLflow model server with a long timeout
#mlflow models serve -m "models:/EmotionRecognizerHF/Staging" -h 0.0.0.0 -p 5001 --no-conda

# Start MLflow model server (serve the model with long timeout)
echo "Starting MLflow model server..."
GUNICORN_CMD_ARGS="--timeout 600" \
mlflow models serve \
  -m "models:/EmotionRecognizerHF/Staging" \
  -h 0.0.0.0 -p 5001 \
  --no-conda