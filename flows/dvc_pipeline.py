from prefect import flow, task
import subprocess

@task
def dvc_add_raw():
    subprocess.run("dvc add data/raw/", shell=True, check=True)

@task
def dvc_preprocess():
    subprocess.run("dvc repro --single-item preprocess", shell=True, check=True)

@task
def dvc_inference():
    subprocess.run("dvc repro --single-item inference", shell=True, check=True)

@task
def dvc_evaluate():
    subprocess.run("dvc repro --single-item evaluate", shell=True, check=True)

@task
def should_retrain(threshold: float = 0.6) -> bool:
    # read the evaluation metrics  and decide
    return False

@task
def mock_train():
    # No real retraining: we just re-register the same model
    subprocess.run("python scripts/mock_train.py", shell=True, check=True)
    subprocess.run("dvc add data/models/emotion_model.pkl", shell=True, check=True)

@task
def dvc_push():
    subprocess.run("dvc push", shell=True, check=True)

@flow(name="dvc_pipeline")
def dvc_pipeline():
    dvc_add_raw()
    dvc_preprocess()
    dvc_metadata()
    dvc_inference()
    if should_retrain():
        mock_train()
    else:
        print("✅ Model OK—skipping retrain")
    dvc_push()

if __name__ == "__main__":
    dvc_pipeline.serve(
        name="dvc_pipeline",
        triggers=[]
    )
