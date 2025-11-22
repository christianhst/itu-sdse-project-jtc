from pathlib import Path
from turtle import pd

from loguru import logger
from tqdm import tqdm
import typer

from customer_classification_model.config import MODELS_DIR, PROCESSED_DATA_DIR



###### REFACTORED CODE FROM MAIN NOTEBOOK ######
#Imports:
import datetime
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split 
from customer_classification_model.data_utils import create_dummy_cols


# Constants used:
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "./artifacts/train_data_gold.csv"
data_version = "00000"
experiment_name = current_date

# MLflow setup:
mlflow.set_experiment(experiment_name)


def load_train_data(path: str) -> pd.DataFrame:
    """
    Load training data from a CSV file.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: The loaded training data.
    """
    data = pd.read_csv(path)
    return data


def data_type_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into categorical and other variables.

    Args:
        data (pd.DataFrame): The input data.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing categorical variables and other variables
    """
    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols]

    other_vars = data.drop(cat_cols, axis=1)

    return cat_vars, other_vars


def one_hot_cat_cols(cat_vars: pd.DataFrame, other_vars: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns and combine with other variables.

    Args:
        cat_vars (pd.DataFrame): The categorical variables.
        other_vars (pd.DataFrame): The other variables.
    
    Returns:
        pd.DataFrame: The combined data with one-hot encoded categorical variables.
    """
    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    data = pd.concat([other_vars, cat_vars], axis=1)

    for col in data:
        data[col] = data[col].astype("float64")
        print(f"Changed column {col} to float")
    return data

def data_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The training and testing sets for features and labels.
    """
    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.15, stratify=y)
    return X_train, X_test, y_train, y_test



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
