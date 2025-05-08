import os
import requests
import json

INFER_URL   = os.getenv("INFERENCE_API_URL", "http://inference:8000/infer")
PRC_DIR     = "data/processed"
OUT_DIR     = "data/inference_output"
LOG_FILE    = "data/metadata/inference_stats.json"
LOG_DIR     = "data/metadata"
AUDIO_NAME  = "audio_metadata.json"

def run_inference():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    run=[]

    # Load metadata
    with open(os.path.join(LOG_DIR, AUDIO_NAME), "r") as f:
        metadata = json.load(f)

    fname = metadata[0]["file"]

    path = os.path.join(PRC_DIR, fname)
    with open(path, "rb") as f:
        files = {"file": (fname, f, "audio/wav")}
        resp = requests.post(INFER_URL, files=files)

    if resp.status_code != 200:
        print(f"Inference failed for {fname}: {resp.text}")
        return

    result = resp.json()
    out_path = os.path.join(OUT_DIR, fname + ".json")
    with open(out_path, "w") as o:
        json.dump(result, o, indent=2)
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
