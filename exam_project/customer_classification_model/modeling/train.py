from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from customer_classification_model.config import MODELS_DIR, PROCESSED_DATA_DIR



###### REFACTORED CODE FROM MAIN NOTEBOOK ######
#Imports:
import datetime
import mlflow


# Constants used:
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "./artifacts/train_data_gold.csv"
data_version = "00000"
experiment_name = current_date

# MLflow setup:
mlflow.set_experiment(experiment_name)


###### DEFAULT CODE BELOW; MODIFY AS NEEDED ######
app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
