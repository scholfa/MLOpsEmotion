#!/bin/bash

echo "Starting Prefect server..."
prefect server start --host 0.0.0.0 --port 4200 &

echo "Waiting for Prefect at http://localhost:4200/health ..."
until curl --silent --fail http://localhost:4200/health; do
  sleep 1
done
echo "Prefect server is healthy."

POOL_NAME="default-pool"
echo "Creating Prefect work-pool '$POOL_NAME'..."
prefect work-pool create --type process "$POOL_NAME"
echo "Work-pool '$POOL_NAME' created."

echo "Verifying work-pool exists..."
until prefect work-pool inspect "$POOL_NAME" >/dev/null 2>&1; do
  echo "Waiting for work pool '$POOL_NAME' to be ready..."
  sleep 2
done
echo "Work pool '$POOL_NAME' is ready."

echo "Starting Prefect worker..."
prefect worker start --pool "$POOL_NAME" &
sleep 5
echo "Prefect worker launched."

echo "Deploying Prefect flow..."
python flows/dvc_pipeline.py &
echo "Prefect flow deployed."

# Keep the container running
wait