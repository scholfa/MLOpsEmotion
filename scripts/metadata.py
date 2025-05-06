import os
import json
import soundfile as sf
import librosa

RAW_DIR = "data/raw"
OUT_FILE = "metadata/audio_metadata.json"

def extract_metadata(raw_dir=RAW_DIR, out_file=OUT_FILE):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    metadata = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        path = os.path.join(raw_dir, fname)

        # 1) Get duration (seconds) with librosa
        duration = librosa.get_duration(filename=path)

        # 2) Get sample rate & channels with soundfile
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


if __name__ == "__main__":
    extract_metadata()
