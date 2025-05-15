#FROM python:3.10-slim
#
## Create working directory
#WORKDIR /app
#
## Install dependencies
#COPY docker/mlflow/requirements.txt .
#COPY scripts/mlflow_init.py .
#RUN pip install --no-cache-dir -r requirements.txt
#
#
## Expose port for MLflow UI
#EXPOSE 5000
#
## Run MLflow server and your script together
#CMD mlflow ui --host 0.0.0.0 & sleep 5 && python mlflow_init.py && tail -f /dev/null
#

FROM python:3.10-slim

# Install bash and curl
RUN apt-get update && apt-get install -y bash curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY docker/mlflow/requirements.txt .
COPY scripts/mlflow_init.py .
COPY scripts/mlflow_emotion_model.py .
COPY docker/mlflow/entrypoint.sh .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports
EXPOSE 5000 5001

# Make entrypoint executable
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

