FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY docker/prefect/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY flows/dvc_pipeline.py flows/dvc_pipeline.py

COPY docker/prefect/configure_dvc_gdrive.sh /usr/local/bin/configure_dvc_gdrive.sh
RUN chmod +x /usr/local/bin/configure_dvc_gdrive.sh

COPY docker/prefect/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

EXPOSE 4200
