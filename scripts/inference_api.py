import io
import os

import requests
import soundfile as sf
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="Emotion Recognition API")
load_dotenv()
MLFLOW_SERVER_URL = os.getenv("MLFLOW_SERVER_URL", "http://mlflow:5001/invocations")


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

    # Prepare the data to be sent to the MLflow model server
    data = {
        "instances": waveform.tolist(),
    }

    # Send the prediction request to MLflow
    response = requests.post(MLFLOW_SERVER_URL, json=data, headers={"Content-Type": "application/json"})
    
    # Debug response
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text[:500]}...")  # Truncate long responses
 
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error from MLflow model inference.")

    prediction = response.json()
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("INFERENCE_PORT", 8000)))
