import io
import os

import requests
import soundfile as sf
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="Emotion Recognition API")
load_dotenv()
MLFLOW_SERVER_URL = os.getenv("MLFLOW_SERVER_URL")


@app.get("/health")
def health_check():
    return {"status": "ok"}


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
#     # Extract features into tensors
#     inputs = feature_extractor(
#         waveform,
#         sampling_rate=SAMPLE_RATE,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#     )
#     # Move tensors to device
#     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
#
#     # Model forward
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     logits = outputs.logits
#     predicted_id = torch.argmax(logits, dim=-1).item()
#     predicted_label = id2label[predicted_id]
#
#     return {"label": predicted_label}

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

    # Prepare the data to be sent to the MLflow model server
    data = {
        "instances": waveform.tolist(),
    }

    # Send the prediction request to MLflow
    response = requests.post(MLFLOW_SERVER_URL, json=data, headers={"Content-Type": "application/json"})

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error from MLflow model inference.")

    prediction = response.json()
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("INFERENCE_PORT", 8000)))
