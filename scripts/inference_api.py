from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import numpy as np
import io
import os

app = FastAPI(title="Emotion Recognition API")

# Load which HF model to use (whisper-based or wav2vec2-based)
HF_MODEL = os.getenv("HF_MODEL_NAME", "superb/wav2vec2-base-superb-ks")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One-time load at startup
feature_extractor = AutoFeatureExtractor.from_pretrained(HF_MODEL)
model = AutoModelForAudioClassification.from_pretrained(HF_MODEL).to(DEVICE)
id2label = model.config.id2label

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(415, "Please upload WAV files only.")

    data = await file.read()
    # Extract features
    inputs = feature_extractor(
        data,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
    ).to(DEVICE)

    # Model forward
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs  = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

    # Return list of (emotion, score)
    response = [
        {"emotion": id2label[i], "score": float(probs[i])}
        for i in range(len(probs))
    ]
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("INFERENCE_PORT", 8000)))
