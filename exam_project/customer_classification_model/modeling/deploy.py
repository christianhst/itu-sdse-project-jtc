import warnings

from mlflow.tracking import MlflowClient

from customer_classification_model.constants import model_name
from customer_classification_model.mlflow_utils import wait_for_deployment

warnings.filterwarnings("ignore")


def deploy_model():
    """
    Deploys the specified model to the staging environment in MLflow Model Registry.
    If the model is not already in the "Staging" stage, it transitions the model
    to "Staging" and waits for the deployment to complete.

    Args:
        None

    Returns:
        None
    """
    model_version = 1
    client = MlflowClient()

    model_version_details = dict(client.get_model_version(name=model_name, version=model_version))
    model_status = True
    if model_version_details["current_stage"] != "Staging":
        client.transition_model_version_stage(
            name=model_name, version=model_version, stage="Staging", archive_existing_versions=True
        )
        model_status = wait_for_deployment(model_name, model_version, "Staging")
        print(model_status)
    else:
        print("Model already in staging")


if __name__ == "__main__":
    deploy_model()
