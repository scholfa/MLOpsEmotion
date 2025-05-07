import os
import librosa
import numpy as np
import soundfile as sf

def preprocess_audio(
    input_dir="data/raw",
    output_dir="data/processed",
    sampling_rate=16000,
    max_duration=30.0,
):
    os.makedirs(output_dir, exist_ok=True)
    file_count = 0

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        file_count += 1
        in_path = os.path.join(input_dir, fname)

        y, sr = librosa.load(in_path, sr=sampling_rate)

        max_len = int(sampling_rate * max_duration)
        if len(y) > max_len:
            y = y[:max_len]
        else:
            y = np.pad(y, (0, max_len - len(y)))

        # Peak‐normalize to [-1.0, 1.0]
        peak = np.max(np.abs(y)) or 1.0
        y = y / peak

        # Write out
        out_path = os.path.join(output_dir, fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, y, sampling_rate)

    print(f"✅ Preprocessed {file_count} files → {output_dir}")

if __name__ == "__main__":
    preprocess_audio()
