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