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

# Copy and install only the extras
COPY docker/inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY /scripts/inference_api.py /app/
EXPOSE 8000

CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
