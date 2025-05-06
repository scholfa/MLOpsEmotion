import streamlit as st
import os
import requests

PREFECT_API_URL = os.getenv("PREFECT_API_URL")
DEPLOYMENT = "dvc-pipeline"
INFERENCE_API = os.getenv("INFERENCE_API_URL", "http://inference:8000/infer")

st.set_page_config(page_title="Emotion Recognition Upload", layout="centered")
st.title("🎤 Upload Audio for Emotion Recognition")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
if uploaded_file:
    save_path = os.path.join("data", "raw", uploaded_file.name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Saved: {uploaded_file.name}")

    if st.button("🚀 Run Emotion Pipeline"):
        resp = requests.post(
            f"{PREFECT_API_URL}/deployments/{DEPLOYMENT}/create_flow_run", json={}
        )
        if resp.status_code == 201:
            st.success("Pipeline triggered!")
        else:
            st.error(f"Trigger failed: {resp.text}")

    if st.button("🔍 Run Instant Inference"):
        files = {"file": open(save_path, "rb")}
        inf = requests.post(INFERENCE_API, files=files).json()
        st.write("Inference result:", inf)
