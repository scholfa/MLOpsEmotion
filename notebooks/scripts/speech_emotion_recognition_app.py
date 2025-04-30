import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import sounddevice as sd
from scipy.io.wavfile import write
import os
import tempfile
import time
import kagglehub

# Set page config
st.set_page_config(
    page_title="Speech Emotion Recognition", 
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match the design in the image
st.markdown("""
<style>
    .main {
        background-color: #121416;
        color: white;
    }
    .input-section {
        border: 2px solid #2a5c2d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: rgba(42, 92, 45, 0.1);
    }
    .output-section {
        border: 2px solid #5c2a2a;
        border-radius: 10px;
        padding: 20px;
        background-color: rgba(92, 42, 42, 0.1);
    }
    .emotion-display {
        text-align: center;
        margin: 20px 0;
    }
    .emoji-icon {
        font-size: 64px;
        margin: 10px 0;
    }
    .stButton > button {
        width: 100%;
        height: 60px;
        background-color: #1e1e1e;
        color: #cccccc;
        border-radius: 10px;
    }
    .stAudio {
        margin-top: 20px;
    }
    /* Center-align the audio controls */
    .stAudio > div {
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the model loading state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Function to load the model
@st.cache_resource
def load_model():
    try:
        model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
        return model, feature_extractor, model.config.id2label
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Load model with a loading spinner
if not st.session_state.model_loaded:
    with st.spinner("Loading emotion recognition model..."):
        model, feature_extractor, id2label = load_model()
        if model is not None:
            st.session_state.model_loaded = True
else:
    model, feature_extractor, id2label = load_model()

# Audio preprocessing function
def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    try:
        audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
        
        max_length = int(feature_extractor.sampling_rate * max_duration)
        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]
        else:
            audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

        inputs = feature_extractor(
            audio_array,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        return inputs, audio_array, sampling_rate
    except Exception as e:
        st.error(f"Error preprocessing audio: {e}")
        return None, None, None

# Emotion prediction function
def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    try:
        inputs, audio_array, sr = preprocess_audio(audio_path, feature_extractor, max_duration)
        if inputs is None:
            return None, None, None
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_label = id2label[predicted_id]
        
        return predicted_label, audio_array, sr
    except Exception as e:
        st.error(f"Error predicting emotion: {e}")
        return None, None, None

# Function to record audio
def record_audio(duration=5, sample_rate=16000):
    # Create a temporary file to store the recording
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Countdown before recording
        for i in range(3, 0, -1):
            status_text.text(f"Recording starts in {i}...")
            time.sleep(1)
        
        # Record audio
        status_text.text("🎙️ Recording... Speak now!")
        
        try:
            # Try to get device info
            device_info = sd.query_devices(sd.default.device[0], 'input')
            sample_rate = int(device_info['default_samplerate'])
        except Exception:
            # If device info is not available, use the provided sample rate
            pass
            
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        
        # Show progress bar during recording
        for i in range(duration):
            progress_bar.progress((i + 1) / duration)
            time.sleep(1)
            
        sd.wait()
        write(temp_file.name, sample_rate, audio)
        status_text.text("✅ Recording completed!")
        progress_bar.empty()
        
        return temp_file.name
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None

# Function to plot audio waveform
def plot_waveform(audio_array, sr):
    try:
        fig, ax = plt.subplots(figsize=(10, 3), facecolor="#1e1e1e")
        
        # Calculate time array
        time = np.arange(0, len(audio_array)) / sr
        
        # Plot waveform
        ax.plot(time, audio_array, color="#c4c4c4")
        
        # Set labels and style
        ax.set_xlabel("time [s]", color="white")
        ax.set_ylabel("db", color="white")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#5c5c5c')
        ax.spines['left'].set_color('#5c5c5c')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_facecolor("#1e1e1e")
        
        # Add waveform icon in the center
        ax_center = (time[-1] / 2, 0)
        ax.annotate('🔊', xy=ax_center, xytext=ax_center, 
                   color='#8c8c8c', fontsize=20, ha='center', va='center',
                   alpha=0.3)
        
        # Tight layout to minimize margins
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error plotting waveform: {e}")
        return None

# Draw emotion icons
def get_emotion_icon(emotion):
    emotion = emotion.lower() if emotion else ""
    if emotion in ["happy", "joy"]:
        return "😊"
    elif emotion in ["sad", "sadness"]:
        return "😢"
    elif emotion in ["angry", "anger"]:
        return "😠"
    elif emotion in ["fear", "fearful"]:
        return "😨"
    elif emotion in ["disgust"]:
        return "🤢"
    elif emotion in ["neutral"]:
        return "😐"
    elif emotion in ["surprise", "surprised"]:
        return "😲"
    else:
        return "🙂"  # Default smiley

# Main app layout
st.title("Speech Emotion Recognition")

# Session state initialization
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'audio_array' not in st.session_state:
    st.session_state.audio_array = None
if 'sr' not in st.session_state:
    st.session_state.sr = None
if 'emotion' not in st.session_state:
    st.session_state.emotion = None

# Input section with green border
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<div style="display: flex; align-items: center;"><div style="writing-mode: vertical-lr; transform: rotate(180deg); margin-right: 15px; font-weight: bold; font-size: 1.2em;">Input</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Record button
    if st.button("🎤 Record", key="record_button", use_container_width=True):
        audio_file = record_audio()
        if audio_file:
            st.session_state.audio_file = audio_file
            
            # Process recording
            with st.spinner("Analyzing emotion..."):
                emotion, audio_array, sr = predict_emotion(
                    audio_file, model, feature_extractor, id2label
                )
                if emotion:
                    st.session_state.audio_array = audio_array
                    st.session_state.sr = sr
                    st.session_state.emotion = emotion
                    st.experimental_rerun()

with col2:
    # Upload button
    uploaded_file = st.file_uploader("🔼 Upload", type=["wav", "mp3", "ogg"])
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            audio_file = temp_file.name
        
        st.session_state.audio_file = audio_file
        
        # Process uploaded audio
        with st.spinner("Analyzing emotion..."):
            emotion, audio_array, sr = predict_emotion(
                audio_file, model, feature_extractor, id2label
            )
            if emotion:
                st.session_state.audio_array = audio_array
                st.session_state.sr = sr
                st.session_state.emotion = emotion
                st.experimental_rerun()

st.markdown('</div></div>', unsafe_allow_html=True)

# Output section with red border
st.markdown('<div class="output-section">', unsafe_allow_html=True)
st.markdown('<div style="display: flex; align-items: center;"><div style="writing-mode: vertical-lr; transform: rotate(180deg); margin-right: 15px; font-weight: bold; font-size: 1.2em;">Output</div>', unsafe_allow_html=True)

# Display waveform if audio has been processed
if st.session_state.audio_array is not None and st.session_state.emotion is not None:
    # Plot and display waveform
    waveform_fig = plot_waveform(st.session_state.audio_array, st.session_state.sr)
    if waveform_fig:
        st.pyplot(waveform_fig)
    
    # Display predicted emotion with emoji
    emotion_icon = get_emotion_icon(st.session_state.emotion)
    st.markdown(f"""
    <div class="emotion-display">
        <p>Predicted emotion</p>
        <div class="emoji-icon">{emotion_icon}</div>
        <p>{st.session_state.emotion}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Audio playback controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.button("▶️", key="play_button", help="Play")
    
    with col2:
        st.button("⏸️", key="pause_button", help="Pause")
    
    with col3:
        st.button("⏹️", key="stop_button", help="Stop")
        
    # Actual audio playback (Streamlit's audio widget has its own controls)
    if st.session_state.audio_file:
        st.audio(st.session_state.audio_file, format="audio/wav")
else:
    st.info("Please record or upload an audio file to see the emotion analysis.")

st.markdown('</div></div>', unsafe_allow_html=True)

# Additional controls at the bottom
if st.session_state.audio_file:
    if st.button("Clear and start over", key="reset_button"):
        st.session_state.audio_file = None
        st.session_state.audio_array = None
        st.session_state.sr = None
        st.session_state.emotion = None
        st.experimental_rerun()

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; color: #888888;">
<small>Speech Emotion Recognition App</small>
</div>
""", unsafe_allow_html=True)