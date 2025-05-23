FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    build-essential \
   && rm -rf /var/lib/apt/lists/*

# Support VS Code Dev Containers / Podman user mapping
RUN useradd -ms /bin/bash vscode

# Install Python dependencies for the app
COPY docker/app/requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy in all application code, scripts, flows, etc.
COPY . .

# Copy and register config dvc script
COPY ../../scripts/configure_dvc_gdrive.sh /usr/local/bin/configure_dvc_gdrive.sh
RUN chmod +x /usr/local/bin/configure_dvc_gdrive.sh

# Install and register your entrypoint script
COPY ../../scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Invoke entrypoint on container start
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Expose ports for Streamlit and Prefect
EXPOSE 8501 4200