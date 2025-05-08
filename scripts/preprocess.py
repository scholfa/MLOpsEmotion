import os
import json
import librosa
import numpy as np
import soundfile as sf

def extract_metadata():
    RAW_DIR = "data/raw"
    LOG_DIR = "data/metadata"
    LOG_NAME = "audio_metadata.json"

    out_file = os.path.join(LOG_DIR, LOG_NAME)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Load existing metadata
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            metadata = json.load(f)
    else:
        raise ValueError("No metadata found. Please run the script with a valid audio file.")

    # Expect metadata to be a dict with 'fname'
    if isinstance(metadata, dict) and "fname" in metadata:
        fname = metadata["fname"]
    else:
        raise ValueError("Invalid metadata format. Expected a dict with 'fname'.")

    path = os.path.join(RAW_DIR, fname)

    # Get duration
    duration = librosa.get_duration(filename=path)

    # Get sample rate and channels
    info = sf.info(path)

    # Add extra metadata
    metadata["duration_sec"] = duration
    metadata["sample_rate"] = info.samplerate
    metadata["channels"] = info.channels

    # Save updated metadata
    with open(out_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Wrote metadata for file: {fname}")

def preprocess_audio():

    IN_DIR="data/raw"
    OUT_DIR="data/processed"
    LOG_DIR="data/metadata"
    AUDIO_NAME="audio_metadata.json"
    SAMPLE_RATE = 16000
    MAX_DUR = 30

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load metadata
    with open(os.path.join(LOG_DIR, AUDIO_NAME), "r") as f:
        metadata = json.load(f)    
    sampling_rate = metadata[0]["sample_rate"]
    fname = metadata[0]["file"]

    # Load and resample
    in_path = os.path.join(IN_DIR, fname)
    y, sr = librosa.load(in_path, sr=SAMPLE_RATE)

    # Trim or pad to max_duration
    max_len = int(sampling_rate * MAX_DUR)
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
