import os, shutil
print("🔄 Mock training: redeploy same model")
os.makedirs("models", exist_ok=True)
shutil.copyfile("models/emotion_model.pkl", "models/emotion_model.pkl")
