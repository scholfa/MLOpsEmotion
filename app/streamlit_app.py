import streamlit as st
import os
import subprocess
import json
import time

# UI setup
st.set_page_config(page_title="Emotion Recognition Upload", layout="centered")
st.title("🎤 Upload Audio for Emotion Recognition")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
if not uploaded_file:
    st.stop()

# Save the file locally
save_dir = os.path.join("data", "raw")
os.makedirs(save_dir, exist_ok=True)

# create a filename with the current timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f"{timestamp}_{uploaded_file.name}"
uploaded_file.name = filename

# Save the uploaded file
save_path = os.path.join(save_dir, uploaded_file.name)
with open(save_path, "wb") as f:
    f.write(uploaded_file.getbuffer())
st.success(f"✅ Saved: {uploaded_file.name}")

# Trigger Prefect pipeline
if st.button("🚀 Run Emotion Pipeline"):
    with st.spinner("Running pipeline..."):
        try:
            subprocess.run("prefect deployment run 'dvc_pipeline/dvc_pipeline'", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            st.error("❌ Failed to trigger pipeline.")
            st.stop()
        st.success("✅ Pipeline run triggered successfully!")

    # Wait for result to contain this filename
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

                matched = next((item for item in results if item["file"] == uploaded_file.name), None)
                if matched:
                    break
            except json.JSONDecodeError:
                pass  # File may be temporarily incomplete during write

        time.sleep(interval)
        elapsed += interval

    if not matched:
        st.error("❌ No matching result found within timeout.")
    else:
        st.success("🎉 Inference complete!")
        st.subheader("📊 Emotion Prediction Result:")
        st.json(matched["result"])
        st.balloons()
