import streamlit as st
import os
import requests

# Config
PREFECT_API_URL = "http://prefect:4200/api"  # Update if running differently
DEPLOYMENT_NAME = "dvc_pipeline/local-dev"

# UI
st.set_page_config(page_title="Emotion Recognition Upload", layout="centered")
st.title("🎤 Upload Audio for Emotion Recognition")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

def save_file(uploaded_file):
    save_path = os.path.join("data", "raw", uploaded_file.name)
    os.makedirs("data/raw", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def trigger_pipeline():
    response = requests.post(
        f"{PREFECT_API_URL}/deployments/{DEPLOYMENT_NAME}/create_flow_run",
        json={}  # You can pass parameters here if needed
    )
    return response

if uploaded_file:
    file_path = save_file(uploaded_file)
    st.success(f"✅ File saved: {uploaded_file.name}")

    if st.button("🚀 Run Emotion Pipeline"):
        st.info("Triggering Prefect flow...")
        try:
            res = trigger_pipeline()
            if res.status_code == 201:
                st.success("✅ Pipeline run created successfully!")
            else:
                st.error(f"❌ Failed to trigger pipeline: {res.text}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
