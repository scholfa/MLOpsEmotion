import io
import os

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import requests
app = FastAPI(title="Emotion Recognition API")

# environment var could be changed for a updated model... default using the same in the testnotebook...
HF_MODEL = os.getenv("HF_MODEL_NAME", "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR="data/models"
MODEL_NAME="emotion_model"
SAMPLE_RATE = 16000

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


# Load models
feature_extractor = AutoFeatureExtractor.from_pretrained(HF_MODEL)
model = AutoModelForAudioClassification.from_pretrained(HF_MODEL).to(DEVICE)
id2label = model.config.id2label

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(status_code=415, detail="Please upload WAV files only.")

    # Read raw bytes and decode to waveform
    audio_bytes = await file.read()
    try:
        waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to decode WAV file.")

    # Extract features into tensors
    inputs = feature_extractor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    # Move tensors to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Model forward
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]

    return {"label": predicted_label}


# MLflow REST API endpoint for predictions
# MLFLOW_SERVER_URL = "http://localhost:5001/invocations"


# @app.post("/infer")
# async def infer(file: UploadFile = File(...)):
#     if file.content_type not in ("audio/wav", "audio/x-wav"):
#         raise HTTPException(status_code=415, detail="Please upload WAV files only.")
#
#     # Read raw bytes and decode to waveform
#     audio_bytes = await file.read()
#     try:
#         waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
#     except Exception:
#         raise HTTPException(status_code=400, detail="Unable to decode WAV file.")
#
#     # Prepare the data to be sent to the MLflow model server
#     data = {
#         "columns": ["waveform"],
#         "data": [waveform.tolist()]
#     }
#
#     # Send the prediction request to MLflow
#     response = requests.post(MLFLOW_SERVER_URL, json=data)
#
#     if response.status_code != 200:
#         raise HTTPException(status_code=500, detail="Error from MLflow model inference.")
#
#     prediction = response.json()
#     return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("INFERENCE_PORT", 8000)))
