import streamlit as st
import os, requests

PREFECT_API_URL    = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")
FLOW_NAME         = "dvc-pipeline"   # the flow name you deployed
DEPLOYMENT_NAME   = "dvc-pipeline"   # the deployment name
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://inference:8000/infer")

@st.cache_data(ttl=600)
def get_deployment_id(flow: str, depl: str) -> str | None:
    """Call Prefect’s `read-deployment-by-name` endpoint to fetch the UUID."""
    url = f"{PREFECT_API_URL}/deployments/name/{flow}/{depl}"
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.json().get("id")
    st.error(f"Could not find deployment via {url}: {resp.status_code} {resp.text}")
    return None

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
    dep_id = get_deployment_id(FLOW_NAME, DEPLOYMENT_NAME)
    if dep_id:
        trigger_url = f"{PREFECT_API_URL}/deployments/{dep_id}/create_flow_run"
        res = requests.post(trigger_url, json={})
        if res.status_code == 201:
            st.success("✅ Pipeline triggered!")
        else:
            st.error(f"❌ Trigger failed: {res.status_code} {res.text}")

# 2) Instant inference
if st.button("🔍 Run Instant Inference"):
    with open(save_path, "rb") as f:
        res = requests.post(INFERENCE_API_URL, files={"file": f})
    if res.ok:
        st.write("Inference result:", res.json())
    else:
        st.error(f"❌ Inference failed: {res.status_code} {res.text}")
