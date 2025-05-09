import streamlit as st
import os
import subprocess
import json
import time
import traceback
import soundfile as sf

# Constants
LOG_DIR = "data/metadata"
RAW_DIR = "data/raw"
LOG_NAME = "metadata.json"
RESULT_FILE = "data/metadata/inference_stats.json"

# UI setup
st.set_page_config(page_title="Emotion Recognition Upload", layout="centered")
st.title("🎤 Upload Audio for Emotion Recognition")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file:
    st.info("📄 File selected. Click the button below to save and run the pipeline.")

    if st.button("🚀 Run Emotion Pipeline"):
        with st.spinner("Processing..."):

            try:
                # Generate filename with timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{timestamp}_{uploaded_file.name}"

                # Save uploaded file
                os.makedirs(RAW_DIR, exist_ok=True)
                raw_path = os.path.join(RAW_DIR, filename)
                with open(raw_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Extract metadata
                # Get size
                file_size = os.path.getsize(raw_path)

                metadata = {
                    "fname": filename,
                    "sname": uploaded_file.name,
                    "size": file_size,
                }

                # Save metadata
                os.makedirs(LOG_DIR, exist_ok=True)
                meta_path = os.path.join(LOG_DIR, LOG_NAME)
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                # check if uploaded file is saved and metadata is written
                if os.path.exists(raw_path) and os.path.exists(meta_path):
                    st.success("✅ File saved and metadata written! Starting inference...")
                    subprocess.run("prefect deployment run 'dvc_pipeline/dvc_pipeline'", shell=True, check=True)
                else:
                    st.error("❌ File or metadata not saved correctly.")
                    st.stop()

            except Exception as e:
                st.error("❌ Something went wrong!")
                st.text(traceback.format_exc())
                st.stop()

        # Wait for result
        timeout = 60  # seconds
        interval = 2
        elapsed = 0
        matched = None

        st.info("⏳ Waiting for inference result...")

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

        if not matched:
            st.error("❌ No matching result found within timeout.")
        else:
            st.success("🎉 Inference complete!")
            st.subheader("📊 Emotion Prediction Result:")
            st.json(matched["result"])
            st.balloons()
else:
    st.stop()
