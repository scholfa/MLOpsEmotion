import io
import os
import soundfile as sf
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

app = FastAPI(title="Emotion Recognition API")

HF_MODEL = os.getenv("HF_MODEL_NAME", "superb/wav2vec2-base-superb-ks")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        sampling_rate=sr,
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("INFERENCE_PORT", 8000)))
