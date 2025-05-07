import streamlit as st
import os
import subprocess

# UI setup
st.set_page_config(page_title="Emotion Recognition Upload", layout="centered")
st.title("🎤 Upload Audio for Emotion Recognition")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
if not uploaded_file:
    st.stop()

# save the file locally
save_dir  = os.path.join("data", "raw")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, uploaded_file.name)
with open(save_path, "wb") as f:
    f.write(uploaded_file.getbuffer())
st.success(f"✅ Saved: {uploaded_file.name}")

# 1) Trigger Prefect flow run
if st.button("🚀 Run Emotion Pipeline"):
    # Trigger the flow run
    run = subprocess.run("prefect deployment run 'dvc_pipeline/dvc_pipeline'",shell=True ,check=True)
    if run.returncode == 0:
        st.success("✅ Pipeline run triggered successfully!")
    else:
        st.error("❌ Failed to trigger pipeline run.")
        st.error(f"Error code: {run.returncode}")
        st.error(f"Error message: {run.stderr.decode()}")