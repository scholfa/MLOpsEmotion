import json
import os
import subprocess
import time
import traceback
import streamlit as st
from audiorecorder import audiorecorder
import wave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Load environment variables
# from dotenv import load_dotenv

# Constants
LOG_DIR = "data/metadata"
RAW_DIR = "data/raw"
LOG_NAME = "metadata.json"
RESULT_FILE = "data/metadata/inference_stats.json"

# UI setup - Changed to wide layout
st.set_page_config(
    page_title="Emotion Recognition", 
    layout="wide",
    page_icon="🎤",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f1f3f5;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: white;
        border-radius: 8px;
        border: 2px solid #e9ecef;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
        border-color: #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
    }
    
    .stError {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header with gradient background
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🎤 Emotion Recognition from Audio - MLOps</h1>
    <p style="color: rgba(255, 255, 255, 0.9); margin-top: 0.5rem; font-size: 1.1rem;">
        Group No. 15 - Fabian Scholpp, Mattia Bosetti, Caitlyn Zuppinger, Patrik Berger
    </p>
</div>
""", unsafe_allow_html=True)

def analyze_audio_amplitude(audio_path):
    """Analyze audio amplitude and create visualization."""
    try:
        # Read WAV file
        with wave.open(audio_path, 'rb') as wav_file:
            # Get audio parameters
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read audio data
            audio_data = wav_file.readframes(n_frames)
        
        # Convert byte data to numpy array
        if sample_width == 2:  # 16-bit audio
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        elif sample_width == 4:  # 32-bit audio
            audio_array = np.frombuffer(audio_data, dtype=np.int32)
        else:
            audio_array = np.frombuffer(audio_data, dtype=np.int8)
        
        # Handle stereo audio (convert to mono)
        if n_channels == 2:
            audio_array = audio_array.reshape(-1, 2).mean(axis=1)
        
        # Normalize audio to [-1, 1]
        audio_array = audio_array.astype(np.float32)
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Create time axis
        duration = n_frames / framerate
        time_axis = np.linspace(0, duration, len(audio_array))
        
        # Calculate amplitude envelope (using moving RMS)
        window_size = int(framerate * 0.05)  # 50ms window
        amplitude_envelope = np.zeros(len(audio_array))
        
        for i in range(len(audio_array)):
            start = max(0, i - window_size // 2)
            end = min(len(audio_array), i + window_size // 2)
            amplitude_envelope[i] = np.sqrt(np.mean(audio_array[start:end]**2))
        
        # Find high amplitude regions (top 30% of max amplitude)
        threshold = np.max(amplitude_envelope) * 0.3
        high_amplitude_mask = amplitude_envelope > threshold
        
        # Create the plot with improved styling
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # Color scheme
        primary_color = '#667eea'
        secondary_color = '#764ba2'
        highlight_color = '#f56565'
        
        # Plot 1: Waveform with highlighted regions
        ax1.plot(time_axis, audio_array, color=primary_color, alpha=0.7, linewidth=0.8)
        ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
        ax1.set_title('Audio Waveform with Important Regions Highlighted', fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(-1.1, 1.1)
        
        # Highlight important regions with better visual
        in_region = False
        start_idx = 0
        
        for i in range(len(high_amplitude_mask)):
            if high_amplitude_mask[i] and not in_region:
                in_region = True
                start_idx = i
            elif not high_amplitude_mask[i] and in_region:
                in_region = False
                ax1.axvspan(time_axis[start_idx], time_axis[i], 
                           alpha=0.2, color=highlight_color, 
                           label='High Energy Region' if start_idx == 0 else "")
        
        # Handle case where region extends to end
        if in_region:
            ax1.axvspan(time_axis[start_idx], time_axis[-1], alpha=0.2, color=highlight_color)
        
        # Add legend for waveform plot
        if np.any(high_amplitude_mask):
            ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gray')
        
        # Plot 2: Amplitude envelope with gradient fill
        ax2.plot(time_axis, amplitude_envelope, color=secondary_color, linewidth=3, label='Amplitude Envelope')
        ax2.axhline(y=threshold, color=highlight_color, linestyle='--', linewidth=2, 
                    alpha=0.7, label=f'Threshold (30% of max)')
        
        # Gradient fill for high amplitude regions
        ax2.fill_between(time_axis, 0, amplitude_envelope, where=high_amplitude_mask, 
                        alpha=0.4, color=highlight_color, label='High Energy Regions')
        ax2.fill_between(time_axis, 0, amplitude_envelope, where=~high_amplitude_mask, 
                        alpha=0.2, color=primary_color)
        
        ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('RMS Amplitude', fontsize=12, fontweight='bold')
        ax2.set_title('Amplitude Envelope Analysis', fontsize=16, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gray')
        ax2.set_ylim(0, max(amplitude_envelope) * 1.1)
        
        # Improve overall appearance
        plt.tight_layout()
        fig.patch.set_facecolor('white')
        
        # Calculate statistics
        total_duration = duration
        high_amplitude_duration = np.sum(high_amplitude_mask) / framerate
        high_amplitude_percentage = (high_amplitude_duration / total_duration) * 100
        
        # Calculate average amplitude in high vs low regions
        avg_high_amplitude = np.mean(amplitude_envelope[high_amplitude_mask]) if np.any(high_amplitude_mask) else 0
        avg_low_amplitude = np.mean(amplitude_envelope[~high_amplitude_mask]) if np.any(~high_amplitude_mask) else 0
        amplitude_ratio = avg_high_amplitude / avg_low_amplitude if avg_low_amplitude > 0 else 0
        
        return fig, {
            "total_duration": total_duration,
            "high_amplitude_duration": high_amplitude_duration,
            "high_amplitude_percentage": high_amplitude_percentage,
            "avg_high_amplitude": avg_high_amplitude,
            "avg_low_amplitude": avg_low_amplitude,
            "amplitude_ratio": amplitude_ratio
        }
        
    except Exception as e:
        st.error(f"Error analyzing audio: {str(e)}")
        return None, None

def process_audio_file(filename, source_name, raw_path):
    """Process an audio file through the emotion recognition pipeline."""
    try:
        # Extract metadata
        file_size = os.path.getsize(raw_path)
        metadata = {
            "file": filename,
            "source": source_name,
            "size": file_size,
        }
        
        # Save metadata
        os.makedirs(LOG_DIR, exist_ok=True)
        meta_path = os.path.join(LOG_DIR, LOG_NAME)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Check if files are saved correctly
        if os.path.exists(raw_path) and os.path.exists(meta_path):
            st.success("✅ File saved and metadata written! Starting inference...")
            subprocess.run("prefect deployment run 'dvc_pipeline/dvc_pipeline'", shell=True, check=True)
        else:
            st.error("❌ File or metadata not saved correctly.")
            return
        
        # Wait for result with progress bar
        timeout = 30  # seconds
        interval = 2
        elapsed = 0
        matched = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.info("⏳ Waiting for inference result...")
        
        while elapsed < timeout:
            if os.path.exists(RESULT_FILE):
                try:
                    with open(RESULT_FILE, "r") as f:
                        results = json.load(f)
                    matched = next((item for item in results if item["file"] == filename), None)
                    if matched:
                        break
                except json.JSONDecodeError:
                    pass  # File may be temporarily incomplete
            
            time.sleep(interval)
            elapsed += interval
            progress = min(elapsed / timeout, 0.95)
            progress_bar.progress(progress)
        
        progress_bar.empty()
        status_text.empty()
        
        if not matched:
            st.error("❌ No matching result found within timeout.")
        else:
            # Extract the result
            emotion_label = matched.get("result", {}).get("predictions", "")
            
            # Convert to icon
            result_em_conv = {
                "angry": ("😡", "Angry", "#ef4444"),
                "disgust": ("🤢", "Disgust", "#84cc16"), 
                "fearful": ("😨", "Fearful", "#8b5cf6"),
                "happy": ("😊", "Happy", "#fbbf24"),
                "neutral": ("😐", "Neutral", "#6b7280"),
                "sad": ("😢", "Sad", "#3b82f6"),
                "surprised": ("😲", "Surprised", "#f97316")
            }
            
            # Get emotion details
            emotion_data = result_em_conv.get(emotion_label[0] if emotion_label else "", ("❓", "Unknown", "#6b7280"))
            result_emoji, emotion_name, emotion_color = emotion_data
            
            # Display results in a nice layout
            st.markdown("---")
            
            # Results section
            col1, col2, col3 = st.columns([2, 3, 2])
            
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, {emotion_color}20 0%, {emotion_color}10 100%); border-radius: 15px; border: 2px solid {emotion_color}40;">
                    <h1 style="font-size: 5rem; margin: 0;">{result_emoji}</h1>
                    <h2 style="color: {emotion_color}; margin: 0.5rem 0; font-size: 2rem;">{emotion_name}</h2>
                    <p style="color: #666; font-size: 1rem;">Emotion detected with AI</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.balloons()
            
            # Add amplitude analysis
            st.markdown("---")
            st.markdown("## 📊 Audio Analysis")
            
            with st.spinner("Analyzing audio characteristics..."):
                fig, stats = analyze_audio_amplitude(raw_path)
                
                if fig and stats:
                    # Display the plot
                    st.pyplot(fig, use_container_width=True)
                    
                    # Display statistics in a nice grid
                    st.markdown("### 📈 Audio Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "🕐 Total Duration", 
                            f"{stats['total_duration']:.2f} sec",
                            help="Total length of the audio recording"
                        )
                    
                    with col2:
                        st.metric(
                            "⚡ High Energy Duration", 
                            f"{stats['high_amplitude_duration']:.2f} sec",
                            help="Duration of high amplitude segments"
                        )
                    
                    with col3:
                        st.metric(
                            "📊 High Energy %", 
                            f"{stats['high_amplitude_percentage']:.1f}%",
                            delta=f"{stats['high_amplitude_percentage']-50:.1f}% vs avg",
                            help="Percentage of recording with high energy"
                        )
                    
                    with col4:
                        st.metric(
                            "📈 Energy Ratio", 
                            f"{stats['amplitude_ratio']:.2f}x",
                            help="Ratio of high to low amplitude regions"
                        )
                    
                    # Interpretation box
                    with st.expander("🔍 How to interpret these results", expanded=True):
                        st.markdown(f"""
                        ### Understanding the Analysis
                        
                        **🔴 Red Highlighted Regions:**
                        - These areas have **higher amplitude** (louder/more energetic speech)
                        - Often contain **stronger emotional expression**
                        - The AI model typically gives these regions **more weight** in emotion detection
                        
                        **📊 Your Recording Analysis:**
                        - **{stats['high_amplitude_percentage']:.1f}%** of your recording has high energy content
                        - High energy segments are **{stats['amplitude_ratio']:.1f}x** louder than quiet parts
                        - The detected emotion **{emotion_name}** is often characterized by {'high' if stats['high_amplitude_percentage'] > 40 else 'moderate to low'} energy patterns
                        
                        **💡 Tips for Better Results:**
                        - Speak clearly and with natural emotion
                        - Avoid background noise
                        - Ensure consistent volume throughout recording
                        """)
            
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        with st.expander("Show error details"):
            st.text(traceback.format_exc())

# Main content area
# Create modern tabs
tab1, tab2, tab3 = st.tabs(["📁 Upload Audio", "🎙️ Record Audio", "ℹ️ About"])

# Tab 1: File Upload - ANGEPASST für zentrierte Ausrichtung
with tab1:
    st.markdown("### Upload your audio file")
    
    # Geändert von [3, 1] auf [1, 2, 1] für Zentrierung
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        uploaded_file = st.file_uploader(
            "Choose a WAV file", 
            type=["wav"],
            help="Select a .wav audio file from your computer"
        )
    
    if uploaded_file:
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.info(f"📄 **Selected file:** {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            # Show audio preview
            st.audio(uploaded_file)
            
            if st.button("🚀 Analyze Emotion", key="upload_btn", use_container_width=True):
                with st.spinner("Processing your audio file..."):
                    try:
                        # Generate filename with timestamp
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"{timestamp}_{uploaded_file.name}"
                        
                        # Save uploaded file
                        os.makedirs(RAW_DIR, exist_ok=True)
                        raw_path = os.path.join(RAW_DIR, filename)
                        with open(raw_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process the file
                        process_audio_file(filename, uploaded_file.name, raw_path)
                        
                    except Exception as e:
                        st.error("❌ Something went wrong!")
                        with st.expander("Show error details"):
                            st.text(traceback.format_exc())

# Tab 2: Audio Recording - Layout bleibt unverändert (bereits zentriert)
with tab2:
    st.markdown("### Record audio directly from your browser")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.info("🎤 Click the button below to start/stop recording")
        
        # Audio recorder
        audio = audiorecorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            key="audio_recorder"
        )
        
        if len(audio) > 0:
            st.markdown("---")
            
            # Display audio player
            st.audio(audio.export().read())
            
            # Show recording info
            duration = len(audio) / 1000  # Convert ms to seconds
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📼 Duration", f"{duration:.2f} seconds")
            with col2:
                st.metric("📊 Sample Rate", "44.1 kHz")
            
            if st.button("🚀 Analyze Emotion", key="record_btn", use_container_width=True):
                with st.spinner("Processing your recording..."):
                    try:
                        # Generate filename with timestamp
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"{timestamp}_recording.wav"
                        
                        # Save recorded audio as WAV file
                        os.makedirs(RAW_DIR, exist_ok=True)
                        raw_path = os.path.join(RAW_DIR, filename)
                        
                        # Export audio to WAV format
                        audio.export(raw_path, format="wav")
                        
                        # Process the file
                        process_audio_file(filename, "recording.wav", raw_path)
                        
                    except Exception as e:
                        st.error("❌ Something went wrong!")
                        with st.expander("Show error details"):
                            st.text(traceback.format_exc())

# Tab 3: About section
with tab3:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### 🎯 About Emotion Recognition
        
        This application uses advanced AI to analyze emotions in speech. It can detect:
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 2rem 0;">
            <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; text-align: center;">
                <span style="font-size: 2rem;">😡</span><br><strong>Angry</strong>
            </div>
            <div style="background: #dcfce7; padding: 1rem; border-radius: 8px; text-align: center;">
                <span style="font-size: 2rem;">🤢</span><br><strong>Disgust</strong>
            </div>
            <div style="background: #e9d5ff; padding: 1rem; border-radius: 8px; text-align: center;">
                <span style="font-size: 2rem;">😨</span><br><strong>Fearful</strong>
            </div>
            <div style="background: #fed7aa; padding: 1rem; border-radius: 8px; text-align: center;">
                <span style="font-size: 2rem;">😊</span><br><strong>Happy</strong>
            </div>
            <div style="background: #e5e7eb; padding: 1rem; border-radius: 8px; text-align: center;">
                <span style="font-size: 2rem;">😐</span><br><strong>Neutral</strong>
            </div>
            <div style="background: #dbeafe; padding: 1rem; border-radius: 8px; text-align: center;">
                <span style="font-size: 2rem;">😢</span><br><strong>Sad</strong>
            </div>
            <div style="background: #ffedd5; padding: 1rem; border-radius: 8px; text-align: center;">
                <span style="font-size: 2rem;">😲</span><br><strong>Surprised</strong>
            </div>
        </div>
        
        ### 🔬 How it works
        
        1. **Upload or Record** - Provide an audio sample
        2. **AI Processing** - Advanced neural networks analyze speech patterns
        3. **Amplitude Analysis** - Identifies high-energy regions that indicate emotion
        4. **Emotion Detection** - Classifies the dominant emotion
        
        ### 📊 Features
        
        - **Real-time Recording** - Record directly from your browser
        - **File Upload** - Analyze existing WAV files
        - **Visual Analysis** - See amplitude patterns and energy distribution
        - **Detailed Statistics** - Understand your audio characteristics
        
        ### 🎯 Best Practices
        
        - Use clear, natural speech
        - Minimize background noise
        - Speak with genuine emotion for best results
        - Recordings of 3-10 seconds work best
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using Streamlit | Emotion Recognition v2.0</p>
    </div>
    """,
    unsafe_allow_html=True
)
