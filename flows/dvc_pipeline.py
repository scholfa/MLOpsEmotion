from prefect import flow, task
import subprocess

@task
def dvc_add_raw():
    subprocess.run("dvc add data/raw/", shell=True, check=True)
    subprocess.run("dvc add data/metadata/metadata.json", shell=True, check=True)

@task
def dvc_preprocess():
    subprocess.run("dvc repro -s preprocess", shell=True, check=True)

@task
def dvc_inference():
    subprocess.run("dvc repro -s inference", shell=True, check=True)

@task
def dvc_evaluate():
    subprocess.run("dvc repro -s evaluate", shell=True, check=True)

@task
def should_retrain(threshold: float = 0.6) -> bool:
    # read the evaluation metrics  and decide
    return False

@task
def mock_train():
    # No real retraining: we just re-register the same model
    subprocess.run("dvc repro -s retrain", shell=True, check=True)

@task
def commit_tag_run(tag: str = "v1.0"):
    # Commit the metadata to Git
    subprocess.run("git add data/metadata", shell=True, check=True)
    subprocess.run("git commit -m 'Add/update metadata metrics' ", shell=True, check=True)
    # subprocess.run(f"git tag {tag}", shell=True, check=True)
    # for now only local commits, to not pollute the repo
    #subprocess.run("git push origin main", shell=True, check=True)     
    #subprocess.run(f"git push origin {tag}", shell=True, check=True)  

@task
def dvc_push():
    subprocess.run("dvc push", shell=True, check=True)

@flow(name="dvc_pipeline")
def dvc_pipeline():
    dvc_add_raw()
    dvc_preprocess()
    dvc_inference()
    dvc_evaluate()
    if should_retrain():
        mock_train()
    else:
        print("✅ Model OK—skipping retrain")
    commit_tag_run()
    dvc_push()

if __name__ == "__main__":
    dvc_pipeline.serve(
        name="dvc_pipeline",
        triggers=[]
    )
