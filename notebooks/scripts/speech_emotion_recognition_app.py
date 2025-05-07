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

plt.rcParams['figure.figsize'] = [10, 3]
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

SAVE_DIR = "saved_audio"
os.makedirs(SAVE_DIR, exist_ok=True)

torch.classes.__path__ = []



# Set page config
st.set_page_config(
    page_title="Speech Emotion Recognition - MLOps", 
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with improved layout and removed colored backgrounds
st.markdown("""
<style>
    body {
        background-color: #121416;
        color: white;
    }

    .main {
        padding: 20px;
    }

    .section-container {
    margin-bottom: 30px;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Label styling */
    .section-label {
        font-weight: bold;
        font-size: 1.2em;
        padding: 12px 20px;
        color: white;
        border-top-left-radius: 12px;
        border-top-right-radius: 12px;
    }

    /* Input + Output Label Colors */
    .input-label {
        background-color: rgba(42, 92, 45, 0.8);
    }

    .output-label {
        background-color: rgba(92, 42, 42, 0.8);
    }

    /* Buttons styling */
    .stButton > button {
        width: 100%;
        height: 60px;
        background-color: #1e1e1e;
        color: #cccccc;
        border-radius: 10px;
        border: 1px solid #333333;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #2a2a2a;
        border-color: #444444;
        transform: translateY(-2px);
    }

    /* File uploader styling */
    .file-uploader {
        background-color: rgba(30, 30, 30, 0.6);
        border-radius: 10px;
        padding: 10px;
    }

    /* Audio player */
    .stAudio {
        margin-top: 20px;
    }

    .stAudio > div {
        display: flex;
        justify-content: center;
    }

    /* Emotion display */
    .emotion-display {
        text-align: center;
        margin: 20px 0;
        padding: 15px;
        background-color: rgba(30, 30, 30, 0.6);
        border-radius: 10px;
    }

    .emoji-icon {
        font-size: 72px;
        margin: 15px 0;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
    }

    .emotion-text {
        font-size: 24px;
        font-weight: 500;
        letter-spacing: 1px;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #888888;
        padding: 10px;
        border-top: 1px solid #333333;
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #2a5c2d !important;
    }

    /* Analysis animation */
    .analysis-status {
        text-align: center;
        padding: 10px;
        margin: 15px 0;
        background-color: rgba(30, 30, 30, 0.6);
        border-radius: 10px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
</style>
""", unsafe_allow_html=True)


# Initialize the model loading state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Add is_analyzing state to track the analysis state
if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False

# Add is_recording state to track recording state
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

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
        # Load the audio file
        audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
        
        # Store the original audio array for visualization
        original_audio = audio_array.copy()
        original_sr = sampling_rate
        
        # Process for the model (padding/truncating to max_duration)
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
        return inputs, original_audio, original_sr
    except Exception as e:
        st.error(f"Error preprocessing audio: {e}")
        return None, None, None

# Emotion prediction function
def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    try:
        inputs, original_audio, original_sr = preprocess_audio(audio_path, feature_extractor, max_duration)
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
        
        # Return the original audio for visualization purposes
        return predicted_label, original_audio, original_sr
    except Exception as e:
        st.error(f"Error predicting emotion: {e}")
        return None, None, None

# Function to record audio with dynamic duration
def record_audio(duration=5, sample_rate=16000):
    # Set recording state to True
    st.session_state.is_recording = True
    
    # Generate unique filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"recording_{timestamp}.wav"
    file_path = os.path.join(SAVE_DIR, filename)
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(3, 0, -1):
            status_text.text(f"Recording starts in {i}...")
            time.sleep(1)
        
        status_text.text("🎙️ Recording in progress... Please speak now!")
        
        try:
            device_info = sd.query_devices(sd.default.device[0], 'input')
            sample_rate = int(device_info['default_samplerate'])
        except Exception:
            pass

        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        
        for i in range(duration):
            progress_bar.progress((i + 1) / duration)
            time.sleep(1)
        
        sd.wait()
        write(file_path, sample_rate, audio)
        status_text.text("✅ Recording complete!")
        progress_bar.empty()
        
        # Recording is complete, set state to False
        st.session_state.is_recording = False
        
        return file_path
    except Exception as e:
        st.error(f"Error during recording: {e}")
        # If there's an error, still set recording state to False
        st.session_state.is_recording = False
        return None

# Function to plot audio waveform with highlighted important regions
def plot_waveform(audio_array, sr):
    try:
        # Calculate actual audio duration in seconds
        audio_duration = len(audio_array) / sr
        
        # Dynamic figure width based on audio duration
        base_width = 5
        base_duration = 5
        width = max(8, min(20, base_width * (audio_duration / base_duration)))
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(width, 1.5), facecolor="#1e1e1e", dpi=300)
        
        # Calculate time array
        time = np.arange(0, len(audio_array)) / sr
        
        # Plot basic waveform
        ax.plot(time, audio_array, color="#c4c4c4", alpha=0.7, linewidth=1.0)
        
        # Identify important regions using energy-based approach
        # Calculate frame-wise energy using a window
        frame_length = int(0.025 * sr)  # 25ms window
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Calculate energy using RMS (root mean square)
        rms = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Convert frames to time
        frames_time = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Normalize RMS to [0, 1]
        rms_norm = rms / np.max(rms) if np.max(rms) > 0 else rms
        
        # Define threshold for "important" regions (adaptive)
        threshold = np.mean(rms_norm) + 0.5 * np.std(rms_norm)
        important_frames = rms_norm > threshold
        
        # Create segments of contiguous important frames
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, is_important in enumerate(important_frames):
            if is_important and not in_segment:
                start_idx = i
                in_segment = True
            elif not is_important and in_segment:
                segments.append((frames_time[start_idx], frames_time[i]))
                in_segment = False
        
        # Add the last segment if still in one
        if in_segment:
            segments.append((frames_time[start_idx], frames_time[-1]))
        
        # Draw the important segments with a different color
        for start, end in segments:
            # Find indices in the original audio array
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, len(audio_array)-1))
            end_idx = max(0, min(end_idx, len(audio_array)-1))
            
            # Draw segment with a different color
            segment_time = time[start_idx:end_idx]
            segment_audio = audio_array[start_idx:end_idx]
            
            if len(segment_time) > 0 and len(segment_audio) > 0:
                # Plot highlighted waveform
                ax.plot(segment_time, segment_audio, color="#ff7043", linewidth=1.8, alpha=0.9)
                
                # Add colored background for the segment
                ax.axvspan(start, end, color="#ff7043", alpha=0.15)
        
        # Add gradient shading under the entire waveform
        ax.fill_between(time, audio_array, alpha=0.1, color="#5c2a2a")
        
        # Add annotation for important regions
        if segments:
            ax.text(0.02, 0.02, "Important regions for emotion recognition", 
                   transform=ax.transAxes, color="#ff7043", fontsize=5,
                   bbox=dict(facecolor="#1e1e1e", alpha=0.7, edgecolor="#ff7043", boxstyle="round,pad=0.2"))
        
        # Set labels and style
        ax.set_xlabel(f"Time [seconds] - Recording duration: {audio_duration:.1f}s", color="white", fontsize=5)
        ax.set_ylabel("Amplitude", color="white", fontsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#5c5c5c')
        ax.spines['left'].set_color('#5c5c5c')
        ax.tick_params(axis='x', colors='white', labelsize=5)
        ax.tick_params(axis='y', colors='white', labelsize=5)
        ax.set_facecolor("#1e1e1e")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.2, color='#5c5c5c')
        
        # Set x-axis limits
        ax.set_xlim(0, audio_duration)
        
        # Adjust x-ticks based on duration
        if audio_duration <= 5:
            ax.set_xticks(np.arange(0, audio_duration + 0.1, 0.5))
        elif audio_duration <= 15:
            ax.set_xticks(np.arange(0, audio_duration + 0.1, 1.0))
        else:
            ax.set_xticks(np.arange(0, audio_duration + 0.1, 2.0))
        
        # Tight layout
        # Compute max amplitude
        max_amp = np.max(np.abs(audio_array))
        # Set soft limit with padding
        ax.set_ylim(-max_amp * 1.2, max_amp * 1.2)
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

# Function to reset the app state
def reset_app_state():
    st.session_state.audio_file = None
    st.session_state.audio_array = None
    st.session_state.sr = None
    st.session_state.emotion = None
    st.session_state.is_analyzing = False
    st.session_state.upload_key = 0

# Session state initialization
if 'first_run' not in st.session_state:
    st.session_state.first_run = True
    
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'audio_array' not in st.session_state:
    st.session_state.audio_array = None
if 'sr' not in st.session_state:
    st.session_state.sr = None
if 'emotion' not in st.session_state:
    st.session_state.emotion = None
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0
    
# Perform reset on first run
if st.session_state.first_run:
    reset_app_state()
    st.session_state.first_run = False

# Main app layout
st.title("Speech Emotion Recognition - MLOps")
st.markdown("<p style='margin-bottom: 10px;'>Upload or record audio to detect the emotion in speech</p>", unsafe_allow_html=True)
st.markdown("<p style='margin-bottom: 25px; font-size: 0.9em; color: #cccccc;'>Available emotions: 😊 Happy/Joy • 😢 Sad/Sadness • 😠 Angry/Anger • 😨 Fear • 🤢 Disgust • 😐 Neutral • 😲 Surprise</p>", unsafe_allow_html=True)

# Input section - Only show if not currently recording
if not st.session_state.is_recording:
    st.markdown("""
    <div class="section-container">
        <div class="section-label input-label">Input</div>
        <div class="input-content">
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎤 Record Audio", key="record_button", use_container_width=True):
            audio_file = record_audio()
            if audio_file:
                st.session_state.audio_file = audio_file
                st.session_state.is_analyzing = True
                st.session_state.upload_key += 1

    with col2:
        uploaded_file = st.file_uploader("🔼 Upload Audio", type=["wav", "mp3", "ogg"], key=st.session_state.upload_key)
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            try:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"upload_{timestamp}{file_extension}"
                file_path = os.path.join(SAVE_DIR, filename)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                st.session_state.audio_file = file_path
                st.session_state.is_analyzing = True

            except Exception as e:
                st.error(f"Error uploading file: {e}")

    st.markdown("</div></div>", unsafe_allow_html=True)
# If recording, show recording status
else:
    # Display recording container
    st.markdown("""
    <div class="section-container">
        <div class="section-label input-label">Recording in Progress</div>
        <div class="input-content">
    """, unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

# Show analysis status if currently analyzing
if st.session_state.is_analyzing and st.session_state.audio_file:
    
    # Process audio with proper error handling
    try:
        with st.spinner("Analyzing emotion..."):
            emotion, audio_array, sr = predict_emotion(
                st.session_state.audio_file, model, feature_extractor, id2label
            )
            
            if emotion and audio_array is not None and sr is not None:
                st.session_state.audio_array = audio_array
                st.session_state.sr = sr
                st.session_state.emotion = emotion
            else:
                st.error("Could not analyze the audio file. Please try a different file.")
    except Exception as e:
        st.error(f"Error during analysis: {e}")
    finally:
        # Always set is_analyzing to False after processing
        st.session_state.is_analyzing = False

# Output section - always show this section
st.markdown("""
<div class="section-container">
    <div class="section-label output-label">Output</div>
    <div class="output-content">
""", unsafe_allow_html=True)

if st.session_state.audio_array is not None and st.session_state.emotion is not None:
    waveform_fig = plot_waveform(st.session_state.audio_array, st.session_state.sr)
    if waveform_fig:
        st.pyplot(waveform_fig)

    emotion_icon = get_emotion_icon(st.session_state.emotion)
    st.markdown(f"""
    <div class="emotion-display">
        <p>Predicted Emotion</p>
        <div class="emoji-icon">{emotion_icon}</div>
        <p class="emotion-text">{st.session_state.emotion.upper()}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='text-align: center; margin-top: 15px;'>Audio Playback</p>", unsafe_allow_html=True)
    cols = st.columns([1, 3, 1])
    with cols[1]:
        if st.session_state.audio_file:
            st.audio(st.session_state.audio_file, format="audio/wav")
else:
    st.info("Please record audio or upload an audio file to see the emotion analysis.")

# Reset Button – now inside the Output area
if st.session_state.audio_file is not None:
    if st.button("🔄 Reset and Start Over", use_container_width=True):
        reset_app_state()
        st.rerun()

st.markdown("</div></div>", unsafe_allow_html=True)