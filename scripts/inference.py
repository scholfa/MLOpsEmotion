import os
import requests
import json

INFER_URL = os.getenv("INFERENCE_API_URL", "http://inference:8000/infer")
RAW_DIR   = "data/processed"
OUT_DIR   = "data/inference_output"
LOG_FILE  = "metadata/inference_log.json"

def run_inference():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    runs = []
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".wav"):
            continue
        path = os.path.join(RAW_DIR, fname)
        with open(path, "rb") as f:
            files = {"file": (fname, f, "audio/wav")}
            resp = requests.post(INFER_URL, files=files)
        if resp.status_code != 200:
            print(f"Inference failed for {fname}: {resp.text}")
            continue
        result = resp.json()
        out_path = os.path.join(OUT_DIR, fname + ".json")
        with open(out_path, "w") as o:
            json.dump(result, o, indent=2)
        runs.append({
            "file": fname,
            "result": result
        })
    with open(LOG_FILE, "w") as o:
        json.dump(runs, o, indent=2)
    print(f"✅ Inference done on {len(runs)} files; log → {LOG_FILE}")

if __name__ == "__main__":
    run_inference()
