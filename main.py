import mlflow


class MLflowWrapper:
    def __init__(self, tracking_uri, experiment_name):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def start_run(self, run_name=None):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=run_name)

    def end_run(self):
        mlflow.end_run()

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_metric(self, key, value):
        mlflow.log_metric(key, value)

    def log_artifact(self, local_path, artifact_path=None):
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model, artifact_path, registered_model_name=None):
        mlflow.sklearn.log_model(
            model, artifact_path, registered_model_name=registered_model_name
        )
