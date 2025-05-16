import os
import random

import mlflow
from mlflow import set_tracking_uri
from mlflow.tracking import MlflowClient
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from mlflow_emotion_model import EmotionRecognitionModel


def mock_mlflow_training():
    losses = []
    epochs = 10
    for epoch in range(1, epochs):
        # Simulate a fake loss decreasing over time
        loss = round(random.uniform(0.7, 1.0) - epoch * 0.1, 4)
        losses.append(loss)
    print("✅ Mock training complete.")
    return losses, epochs


# Set the tracking URI (ensure this is where MLflow is running)
set_tracking_uri("http://localhost:5000")

# Create an MLflow client
client = MlflowClient()

# Provide an Experiment description and tags
experiment_description = "Emotion recognition experiment "
experiment_tags = {
    "project_name": "emotion-recognition",
    "team": "stores-ml",
    "project_quarter": "Q1-2025",
    "mlflow.note.content": experiment_description,
}

# Create the experiment with the tags
experiment_name = "Emotion_Recognition_Models"
if not client.get_experiment_by_name(experiment_name):
    client.create_experiment(
        name=experiment_name, tags=experiment_tags
    )
else:
    print(f"Experiment '{experiment_name}' already exists.")

# Load model
print("Downloading model from Hugging Face...")
HF_MODEL = os.getenv("HF_MODEL_NAME", "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")
MODEL_DIR = "data/models"
MODEL_NAME = "emotion_model"

# try to load local model
if not os.path.exists(MODEL_DIR):
    print(f"Model directory {MODEL_DIR} not found. Creating...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Model directory {MODEL_DIR} created.")
else:
    print(f"Model directory {MODEL_DIR} already exists.")

# Check if the model is already downloaded
if not os.path.exists(os.path.join(MODEL_DIR, MODEL_NAME)):
    print(f"Model {MODEL_NAME} not found in {MODEL_DIR}. Downloading...")
    # Download the model
    AutoModelForAudioClassification.from_pretrained(HF_MODEL).save_pretrained(MODEL_DIR)
    print(f"Model {MODEL_NAME} downloaded and saved to {MODEL_DIR}.")
else:
    print(f"Model {MODEL_NAME} already exists in {MODEL_DIR}. Loading...")
    # Load the model
    AutoModelForAudioClassification.from_pretrained(MODEL_DIR).save_pretrained(MODEL_DIR)
    print(f"Model {MODEL_NAME} loaded from {MODEL_DIR}.")

print("Downloading feature extractor from Hugging Face...")
feature_extractor = AutoFeatureExtractor.from_pretrained(HF_MODEL).save_pretrained(MODEL_DIR)

print("Model and feature extractor saved locally.")

mlflow.set_experiment(experiment_name)

mlflow.autolog(log_models=True)

with mlflow.start_run(run_name="register-whisper-emotion") as run:
    print("Starting mock training loop...")
    loss, epochs = mock_mlflow_training()
    for l, e in zip(loss, range(1, epochs)):
        mlflow.log_metric("loss", f"{l:2f}", step=e)
    print("Mock training complete.")

    mlflow.pyfunc.log_model(
        artifact_path="emotion_model",
        python_model=EmotionRecognitionModel(),
        artifacts={"model_dir": MODEL_DIR},
        registered_model_name="EmotionRecognizerHF"
    )
    print("Model logged and registered with MLflow using pyfunc interface.")

# Promote the registered model version to 'Staging'
latest_version = client.get_latest_versions("EmotionRecognizerHF", stages=["None"])[0].version
client.transition_model_version_stage(
    name="EmotionRecognizerHF",
    version=latest_version,
    stage="Staging"
)
print(f"Model version {latest_version} promoted to 'Staging'")
