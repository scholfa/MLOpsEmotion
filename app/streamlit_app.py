import streamlit as st
import os
import requests

# Configuration
PREFECT_API_URL    = os.getenv("PREFECT_API_URL")
DEPLOYMENT_NAME   = "dvc-pipeline"
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://inference:8000/infer")

@st.cache_data(ttl=600)
def get_deployment_id(name: str) -> str | None:
    """Fetch all deployments and return the UUID for the one matching `name`."""
    try:
        resp = requests.get(f"{PREFECT_API_URL}/deployments")
        resp.raise_for_status()
        for dep in resp.json().get("deployments", []):
            if dep.get("name") == name:
                return dep["id"]
    except Exception as e:
        st.error(f"Error fetching Prefect deployments: {e}")
    return None

# UI setup
st.set_page_config(page_title="Emotion Recognition Upload", layout="centered")
st.title("🎤 Upload Audio for Emotion Recognition")

# File uploader
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
if uploaded_file:
    save_dir  = os.path.join("data", "raw")
    save_path = os.path.join(save_dir, uploaded_file.name)
    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Saved: {uploaded_file.name}")

    # Trigger Prefect flow
    if st.button("🚀 Run Emotion Pipeline"):
        dep_id = get_deployment_id(DEPLOYMENT_NAME)
        if not dep_id:
            st.error(f"Deployment '{DEPLOYMENT_NAME}' not found.")
        else:
            run_resp = requests.post(
                f"{PREFECT_API_URL}/deployments/{dep_id}/create_flow_run",
                json={}
            )
            if run_resp.status_code == 201:
                st.success("✅ Pipeline triggered!")
            else:
                st.error(f"❌ Trigger failed: {run_resp.text}")

    # Instant inference via API
    if st.button("🔍 Run Instant Inference"):
        with open(save_path, "rb") as f:
            files = {"file": f}
            inf_resp = requests.post(INFERENCE_API_URL, files=files)
        if inf_resp.status_code == 200:
            st.write("Inference result:", inf_resp.json())
        else:
            st.error(f"❌ Inference failed: {inf_resp.text}")
