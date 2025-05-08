import os
import json
import mlflow

def evaluate_model():
    MODEL_DIR="data/models"
    MODEL_NAME="emotion_model"
    LOG_DIR="data/metadata"
    LOG_NAME="eval_metrics.json"
    
    # create dummy json output
    os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, LOG_NAME)
    with open(LOG_FILE, "w") as f:
        json.dump({"model": LOG_NAME}, f, indent=2)
    print(f"✅ Wrote log to {LOG_FILE}")

if __name__=="__main__":
    evaluate_model()
