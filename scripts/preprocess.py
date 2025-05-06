import os
import librosa
import numpy as np

def preprocess_audio(
    input_dir="data/raw",
    output_dir="data/processed",
    sampling_rate=16000,
    max_duration=30.0,
):
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".wav"):
            continue
        in_path = os.path.join(input_dir, fname)

        # 1) Load & resample
        y, sr = librosa.load(in_path, sr=sampling_rate)

        # 2) Pad / truncate
        max_len = int(sampling_rate * max_duration)
        if len(y) > max_len:
            y = y[:max_len]
        else:
            y = np.pad(y, (0, max_len - len(y)))

        # 3) Write out normalized file
        out_path = os.path.join(output_dir, fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        librosa.output.write_wav(out_path, y, sampling_rate)

    print(f"✅ Preprocessed {len(os.listdir(input_dir))} files → {output_dir}")


if __name__ == "__main__":
    preprocess_audio()
