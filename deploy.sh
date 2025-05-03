#!/bin/bash

echo "🔧 Building and applying Prefect deployment for dvc_pipeline..."

# Build the deployment
prefect deployment build flows/dvc_pipeline.py:dvc_pipeline \
    --name local-dev \
    --work-queue default \
    --tag local \
    --output dvc_pipeline-deployment.yaml

# Apply the deployment to Prefect
prefect deployment apply dvc_pipeline-deployment.yaml

echo "✅ Deployment applied! Visit http://localhost:4200 to trigger or monitor runs."