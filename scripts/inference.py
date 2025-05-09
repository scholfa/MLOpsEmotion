import os
import requests
import json

# all these things could be part of env vars
INFER_URL   = os.getenv("INFERENCE_API_URL", "http://inference:8000/infer")
PRC_DIR     = "data/processed"
LOG_FILE    = "data/metadata/inference_stats.json"
LOG_DIR     = "data/metadata"
META_DATA   = "metadata.json"

def run_inference():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    run=[]

    # Load metadata
    with open(os.path.join(LOG_DIR, META_DATA), "r") as f:
        metadata = json.load(f)

    fname = metadata["file"]

    path = os.path.join(PRC_DIR, fname)
    with open(path, "rb") as f:
        files = {"file": (fname, f, "audio/wav")}
        resp = requests.post(INFER_URL, files=files)

    if resp.status_code != 200:
        print(f"Inference failed for {fname}: {resp.text}")
        return

    result = resp.json()
    run.append({
        "file": fname,
        "result": result
    })       

    # Save inference stats
    with open(LOG_FILE, "w") as o:
        json.dump(run, o, indent=2)

    print(f"✅ Inference done")

if __name__ == "__main__":
    run_inference()
