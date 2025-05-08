import os
import json
import librosa
import numpy as np
import soundfile as sf

def extract_metadata():

    RAW_DIR = "data/raw"
    LOG_DIR="data/metadata"
    LOG_NAME="audio_metadata.json"

    out_file = os.path.join(LOG_DIR, LOG_NAME)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    metadata = []
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.lower().endswith(".wav"):
            continue
        path = os.path.join(RAW_DIR, fname)

        # Get duration (seconds) with librosa
        duration = librosa.get_duration(filename=path)

        # Get sample rate & channels with soundfile
        info = sf.info(path)
        metadata.append({
            "file": fname,
            "duration_sec": duration,
            "sample_rate": info.samplerate,
            "channels": info.channels
        })

    with open(out_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Wrote metadata for {len(metadata)} files to {out_file}")


def preprocess_audio():

    IN_DIR="data/raw"
    OUT_DIR="data/processed"
    LOG_DIR="data/metadata"
    AUDIO_NAME="audio_metadata.json"
    SAMPLE_RATE = 16000

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load metadata
    with open(os.path.join(LOG_DIR, AUDIO_NAME), "r") as f:
        metadata = json.load(f)    
    sampling_rate = metadata[0]["sample_rate"]
    fname = metadata[0]["file"]
    max_duration = metadata[0]["duration_sec"]

    # Load and resample
    in_path = os.path.join(IN_DIR, fname)
    y, sr = librosa.load(in_path, sr=SAMPLE_RATE)

    # Trim or pad to max_duration
    max_len = int(sampling_rate * max_duration)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    # Peak-normalize to [-1.0, 1.0]
    peak = np.max(np.abs(y)) or 1.0
    y = y / peak

    # Write processed file
    out_path = os.path.join(OUT_DIR, fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, y, sampling_rate)

    print(f"✅ Preprocessed")

if __name__ == "__main__":
    extract_metadata()
    preprocess_audio()
