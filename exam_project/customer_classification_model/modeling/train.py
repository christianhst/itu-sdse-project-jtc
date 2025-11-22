from pathlib import Path
import json  # used for data drift
import pandas as pd  # used for data drift

from loguru import logger
from tqdm import tqdm
import typer

from customer_classification_model.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def save_data_drift(data: pd.DataFrame, schema_path: Path, training_data_path: Path) -> None:
    """Save data drift schema and training dataset artifacts.

    Args:
        data (pd.DataFrame): DataFrame containing the data (cont + cat)
        schema_path (Path): Output path for the schema JSON file.
        training_data_path (Path): Output path for the training CSV file.

    Returns:
        None
    """

    # Save schema (list of column names)
    data_columns = list(data.columns)
    with open(schema_path, "w") as f:
        json.dump(data_columns, f)

    # Save full training data
    data.to_csv(training_data_path, index=False)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    schema_path: Path = MODELS_DIR / "columns_drift.json",
    training_data_path: Path = MODELS_DIR / "training_data.csv",
    # -----------------------------------------
):
    # Load data
    logger.info("Loading data...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)

    # Combine features and labels for data drift monitoring
    combined_data = X.copy()
    combined_data["label"] = y

    # Saave schema for data drift monitoring
    save_data_drift(
        data=combined_data,
        schema_path=schema_path,
        training_data_path=training_data_path,
    )

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
