# Makefile for MLOpsEmotion (Podman-Compose)

# Change if you use a different compose wrapper
COMPOSE       := podman-compose
PROJECT_NAME  := mlopsemotion

.PHONY: all build-app build-inference build-mlflow build up down logs status \
        preprocess metadata inference evaluate dvc-repro dvc-push \
        pipeline run-pipeline clean

# Default: build both images and bring everything up
all: build up

# ─── Build images ──────────────────────────────────────────────────────────────

build-app:
	$(COMPOSE) build app

build-inference:
	$(COMPOSE) build inference

build-mlflow:
	$(COMPOSE) build mlflow

build: build-app build-inference build-mlflow

# ─── Compose lifecycle ────────────────────────────────────────────────────────

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

status:
	$(COMPOSE) ps

# ─── DVC stages (local, without Prefect) ───────────────────────────────────────

preprocess:
	python scripts/preprocess.py

metadata:
	python scripts/metadata.py

inference:
	python scripts/inference.py

evaluate:
	python scripts/evaluate_model.py

# Run full DVC pipeline (all stages in dvc.yaml)
dvc-repro:
	dvc repro

# Push DVC artifacts to remote
dvc-push:
	dvc push

# ─── Prefect / DVC pipeline (via Python flow) ─────────────────────────────────

# Trigger your Prefect-orchestrated DVC pipeline
run-pipeline:
	python flows/dvc_pipeline.py

# Alias for “build → up → run pipeline”
pipeline: build up run-pipeline

# ─── Cleanup ───────────────────────────────────────────────────────────────────

# Tear down containers, remove built images and volumes
clean: down
	# Remove the two built images (ignore errors if they don't exist)
	podman image rm mlopsemotion-app mlopsemotion-inference mlopsemotion-mlflow || true
	# Remove the named volumes (ignore errors if they don't exist)
	podman volume rm ${PROJECT_NAME}_app-data ${PROJECT_NAME}_app-shared || true
