import mlflow
def evaluate_model():
    with mlflow.start_run(run_name="eval_run"):
        mlflow.log_metric("f1_score", 0.58)
if __name__=="__main__":
    evaluate_model()
