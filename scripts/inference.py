import os
import json
from datetime import datetime
from hashlib import sha256

def fake_predict(path):
    return {"start": 0.0, "end": 3.0, "emotion": "happy"}

def hash_file(path):
    with open(path, "rb") as f:
        return sha256(f.read()).hexdigest()

def run():
    os.makedirs("data/inference_output", exist_ok=True)
    os.makedirs("metadata", exist_ok=True)
    log = []

    for fname in os.listdir("data/processed"):
        if not fname.endswith(".wav"):
            continue
        pred = fake_predict(fname)
        with open(f"data/inference_output/{fname}.json", "w") as f:
            json.dump(pred, f)
        log.append({
            "file": fname,
            "dvc_hash": hash_file(f"data/processed/{fname}"),
            "prediction": pred,
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": "v1.0",
            "user_feedback": None
        })

    with open("metadata/inference_log.json", "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    run()
