from prefect import flow, task
import subprocess
from mlflow.tracking import MlflowClient

@task
def dvc_add_raw():
    subprocess.run("dvc add data/raw/", shell=True, check=True)

@task
def dvc_repro():
    subprocess.run("dvc repro", shell=True, check=True)

@task
def should_retrain(threshold=0.6):
    client = MlflowClient()
    run = client.search_runs(["0"], order_by=["start_time DESC"], max_results=1)[0]
    return run.data.metrics.get("f1_score", 1.0) < threshold

@task
def mock_train():
    subprocess.run(
        "dvc run -n mock_train -d data/processed/ -d scripts/mock_train.py "
        "-o models/emotion_model.pkl python scripts/mock_train.py",
        shell=True,
        check=True
    )

@flow
def dvc_pipeline():
    dvc_add_raw()
    dvc_repro()
    if should_retrain():
        mock_train()
    else:
        print("✅ Model performance acceptable – skipping retrain")

if __name__ == "__main__":
    dvc_pipeline()