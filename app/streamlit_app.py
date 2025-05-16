import json
import os
import subprocess
import time
import traceback
import streamlit as st

# Load environment variables
# from dotenv import load_dotenv

# Constants
LOG_DIR = "data/metadata"
RAW_DIR = "data/raw"
LOG_NAME = "metadata.json"
RESULT_FILE = "data/metadata/inference_stats.json"

# Load environment variables
# load_dotenv()
# PREFECT_API = os.getenv("PREFECT_API_NET")
# DEPLOYMENT_ID = os.getenv("DEPLOYMENT_ID")

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
                    "file": filename,
                    "source": uploaded_file.name,
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
                    # print("POST request to Prefect API")
                    # print("ENDPOINT", f"{PREFECT_API}/deployments/name/{DEPLOYMENT_ID}/run")
                    # response = requests.post(
                    #     # f"{PREFECT_API}/deployments/name/{DEPLOYMENT_ID}/run",
                    #     f"{PREFECT_API}/deployments/0f698b89-9bb7-4fa9-a501-c078b64fdec0/create_flow_run",
                    #     timeout=10,
                    #     body={{"state": {"type": "SCHEDULED", "message": "Run from the Prefect UI with defaults",
                    #                      "state_details": {}}}}
                    # )
                    # print("ENDED", response.json())
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
            # extract the result
            emotion_label = matched.get("result", {}).get("predictions", "")

            # convert to icon
            result_em_conv = {
                "angry": "😡",
                "disgust": "🤢",
                "fearful": "😨",
                "happy": "😊",
                "neutral": "😐",
                "sad": "😢",
                "surprised": "😲"
            }
            # convert to emoji
            result_emoji = result_em_conv.get(emotion_label[0], "❓")

            st.success("🎉 Inference complete!")
            st.subheader(f"📊 Emotion Prediction Result: {result_emoji} {emotion_label}")
            st.balloons()
else:
    st.stop()
