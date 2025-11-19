import time

from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking.client import MlflowClient


def wait_until_ready(model_name, model_version):
    """Poll the MLflow Model Registry until a model version becomes ready.

    Args:
        model_name (str): Registered model name.
        model_version (str | int): Model version identifier to monitor.
    """
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
            name=model_name,
            version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)


def wait_for_deployment(model_name, model_version, stage="Staging"):
    """Block until a model version reaches the specified deployment stage.

    Args:
        model_name (str): Registered model name.
        model_version (str | int): Model version identifier.
        stage (str, optional): Desired stage such as ``"Staging"`` or ``"Production"``.
            Defaults to ``"Staging"``.

    Returns:
        bool: ``True`` after the model reaches the target stage.
    """
    client = MlflowClient()
    status = False
    while not status:
        model_version_details = dict(
            client.get_model_version(name=model_name, version=model_version)
        )
        if model_version_details["current_stage"] == stage:
            print(f"Transition completed to {stage}")
            status = True
            break
        else:
            time.sleep(2)
    return status
