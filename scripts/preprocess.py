import os
import torchaudio

def preprocess_audio(input_dir="data/raw", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.endswith(".wav"):
            waveform, sr = torchaudio.load(os.path.join(input_dir, fname))
            waveform = waveform / waveform.abs().max()
            torchaudio.save(os.path.join(output_dir, fname), waveform, sr)

if __name__ == "__main__":
    preprocess_audio()
