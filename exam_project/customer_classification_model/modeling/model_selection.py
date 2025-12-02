import json

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from customer_classification_model.constants import artifact_path, experiment_name, model_name


def getting_experiment_and_best_model_results():
    experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids, order_by=["metrics.f1_score DESC"], max_results=1
    ).iloc[0]

    with open("./artifacts/model_results.json", "r") as f:
        model_results = json.load(f)
        results_df = pd.DataFrame({
            model: val["weighted avg"] for model, val in model_results.items()
        }).T

    best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
    print(f"Best model: {best_model}")

    return experiment_best, best_model


def get_production_model():
    client = MlflowClient()
    prod_model = [
        model
        for model in client.search_model_versions(f"name='{model_name}'")
        if dict(model)["current_stage"] == "Production"
    ]
    prod_model_exists = len(prod_model) > 0

    if prod_model_exists:
        prod_model_version = dict(prod_model[0])["version"]
        prod_model_run_id = dict(prod_model[0])["run_id"]

        print("Production model name: ", model_name)
        print("Production model version:", prod_model_version)
        print("Production model run id:", prod_model_run_id)

    else:
        print("No model in production")
