# main.pyのテストを作成する
# mlflow_wrapper classのテストを書きます
# mlflowはmockして叩かないようにしたいです。
from main import MLflowWrapper
from unittest.mock import MagicMock
import pytest
import mlflow
from mlflow.sklearn import log_model


class TestMLflowWrapper:
    def test_init(self):
        mlflow_wrapper = MLflowWrapper("tracking_uri", "experiment_name")
        assert mlflow_wrapper.tracking_uri == "tracking_uri"
        assert mlflow_wrapper.experiment_name == "experiment_name"

    def test_start_run(self):
        mlflow_wrapper = MLflowWrapper("tracking_uri", "experiment_name")
        mlflow.set_tracking_uri = MagicMock()
        mlflow.set_experiment = MagicMock()
        mlflow.start_run = MagicMock()
        mlflow_wrapper.start_run()
        mlflow.set_tracking_uri.assert_called_once_with("tracking_uri")
        mlflow.set_experiment.assert_called_once_with("experiment_name")
        mlflow.start_run.assert_called_once_with(run_name=None)

    def test_end_run(self):
        mlflow_wrapper = MLflowWrapper("tracking_uri", "experiment_name")
        mlflow.end_run = MagicMock()
        mlflow_wrapper.end_run()
        mlflow.end_run.assert_called_once_with()

    def test_log_param(self):
        mlflow_wrapper = MLflowWrapper("tracking_uri", "experiment_name")
        mlflow.log_param = MagicMock()
        mlflow_wrapper.log_param("key", "value")
        mlflow.log_param.assert_called_once_with("key", "value")

    def test_log_metric(self):
        mlflow_wrapper = MLflowWrapper("tracking_uri", "experiment_name")
        mlflow.log_metric = MagicMock()
        mlflow_wrapper.log_metric("key", "value")
        mlflow.log_metric.assert_called_once_with("key", "value")

    def test_log_artifact(self):
        mlflow_wrapper = MLflowWrapper("tracking_uri", "experiment_name")
        mlflow.log_artifact = MagicMock()
        mlflow_wrapper.log_artifact("local_path", "artifact_path")
        mlflow.log_artifact.assert_called_once_with("local_path", "artifact_path")

    def test_log_model(self):
        mlflow_wrapper = MLflowWrapper("tracking_uri", "experiment_name")
        mlflow.sklearn.log_model = MagicMock()
        mlflow_wrapper.log_model("model", "artifact_path", "registered_model_name")
        mlflow.sklearn.log_model.assert_called_once_with(
            "model", "artifact_path", registered_model_name="registered_model_name"
        )
