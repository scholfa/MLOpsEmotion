import streamlit as st
import os
import subprocess
import json
import time
import traceback

LOG_DIR = "data/metadata"
LOG_NAME = "audio_metadata.json"

# UI setup
st.set_page_config(page_title="Emotion Recognition Upload", layout="centered")
st.title("🎤 Upload Audio for Emotion Recognition")

# Handle file upload
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file:
    if "saved_filename" not in st.session_state:
        # Generate timestamped filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        new_filename = f"{timestamp}_{uploaded_file.name}"

        # Save uploaded file
        save_dir = os.path.join("data", "raw")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, new_filename)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Saved: {new_filename}")

        # Write metadata
        metadata = {"fname": new_filename}
        os.makedirs(LOG_DIR, exist_ok=True)
        file_path = os.path.join(LOG_DIR, LOG_NAME)
        try:
            with open(file_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            st.write("✅ Metadata written.")
        except Exception as e:
            st.error(f"Error writing metadata: {e}")
            st.text(traceback.format_exc())

        # Store filename in session state
        st.session_state.saved_filename = new_filename
    else:
        st.success(f"📄 File already uploaded: {st.session_state.saved_filename}")
else:
    st.stop()

# Trigger Prefect pipeline
if st.button("🚀 Run Emotion Pipeline"):
    with st.spinner("Running pipeline..."):
        try:
            subprocess.run("prefect deployment run 'dvc_pipeline/dvc_pipeline'", shell=True, check=True)
        except subprocess.CalledProcessError:
            st.error("❌ Failed to trigger pipeline.")
            st.stop()
        st.success("✅ Pipeline run triggered successfully!")

    # Wait for result
    result_file = "data/metadata/inference_stats.json"
    timeout = 60  # seconds
    interval = 2
    elapsed = 0
    matched = None

    st.info("⏳ Waiting for inference result...")

    while elapsed < timeout:
        if os.path.exists(result_file):
            try:
                with open(result_file, "r") as f:
                    results = json.load(f)

                matched = next((item for item in results if item["file"] == st.session_state.saved_filename), None)
                if matched:
                    break
            except json.JSONDecodeError:
                pass  # File may be temporarily incomplete

        time.sleep(interval)
        elapsed += interval

    if not matched:
        st.error("❌ No matching result found within timeout.")
    else:
        st.success("🎉 Inference complete!")
        st.subheader("📊 Emotion Prediction Result:")
        st.json(matched["result"])
        st.balloons()

# Reset button to allow new upload
if "saved_filename" in st.session_state:
    if st.button("🔄 Upload Another File"):
        del st.session_state.saved_filename
        st.experimental_rerun()
