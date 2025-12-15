from datetime import datetime
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from customer_classification_model.data_utils import (
    create_directories,
    describe_numeric_col,
    impute_missing_values,
    load_data,
)

warnings.filterwarnings("ignore")

# Date limits for data
max_date = "2024-01-31"
min_date = "2024-01-01"


def time_limit_data(data: pd.DataFrame, max_date=max_date, min_date=min_date) -> None:
    """Limit data to a specific date range and save the limits to a JSON file.

    Args:
        data (pd.DataFrame): DataFrame containing a 'date_part' column.
        max_date (str): Maximum date as a string in 'YYYY-MM-DD' format.
        min_date (str): Minimum date as a string in 'YYYY-MM-DD' format.
    """
    if not max_date:
        max_date = pd.to_datetime(datetime.now().date()).date()
    else:
        max_date = pd.to_datetime(max_date).date()

    min_date = pd.to_datetime(min_date).date()

    # Time limit data
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

    min_date = data["date_part"].min()
    max_date = data["date_part"].max()
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
    with open("./artifacts/date_limits.json", "w") as f:
        json.dump(date_limits, f)


def feature_selection(data: pd.DataFrame) -> pd.DataFrame:
    """Select features by removing specific columns from the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with selected features.
    """

    # Removing columns that will be added back after the EDA
    data = data.drop(
        [
            "is_active",
            "marketing_consent",
            "first_booking",
            "existing_customer",
            "last_seen",
            "domain",
            "country",
            "visited_learn_more_before_booking",
            "visited_faq",
        ],
        axis=1,
    )
    return data


def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by handling missing values and filtering.

    Args:
        data (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """

    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)

    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])

    data = data[data.source == "signup"]

    return data


def create_cat_cols(data: pd.DataFrame) -> pd.DataFrame:
    """Convert specified columns to categorical (object) type.

    Args:
        data (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with categorical columns converted to object type.
    """
    vars = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]

    for col in vars:
        data[col] = data[col].astype("object")
    return data


def separate_cat_and_cont_cols(data: pd.DataFrame) -> pd.DataFrame:
    """Separate continuous and categorical columns in the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Tuple containing continuous and categorical DataFrames.
    """

    cont_vars = data.loc[:, ((data.dtypes == "float64") | (data.dtypes == "int64"))]
    cat_vars = data.loc[:, (data.dtypes == "object")]

    return cont_vars, cat_vars


def outliers(cont_vars: pd.DataFrame, z: float = 2.0) -> pd.DataFrame:
    """Detect outliers in continuous variables using +-z standard deviations.

    Args:
        cont_vars (pd.DataFrame): continuous variables.
        z (float): Number of standard deviations for clipping.

    Returns:
        pd.DataFrame: DataFrame with outliers clipped.
    """
    cont_vars = cont_vars.apply(
        lambda x: x.clip(lower=x.mean() - z * x.std(), upper=x.mean() + z * x.std())
    )
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv("./artifacts/outlier_summary.csv")

    return cont_vars


def impute_continuous(cont_vars: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in continuous columns.

    Args:
        cont_vars (pd.DataFrame): DataFrame containing continuous variables.

    Returns:
        pd.DataFrame: DataFrame with missing continuous values imputed.
    """
    cont_vars = cont_vars.apply(impute_missing_values, method="mean")
    return cont_vars


def impute_categorical(cat_vars: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in categorical columns.

    Args:
        cat_vars (pd.DataFrame): DataFrame containing categorical variables.

    Returns:
        pd.DataFrame: DataFrame with missing categorical values imputed.
    """
    if "customer_code" in cat_vars.columns:
        cat_vars.loc[cat_vars["customer_code"].isna(), "customer_code"] = "None"

    cat_vars = cat_vars.apply(impute_missing_values)
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv("./artifacts/cat_missing_impute.csv")
    return cat_vars


def standardize_continuous(
    cont_vars: pd.DataFrame, scaler_path: str = "./artifacts/scaler.pkl"
) -> pd.DataFrame:
    """Standardize continuous variables using MinMaxScaler and save the scaler.

    Args:
        cont_vars (pd.DataFrame): DataFrame containing continuous variables.
        scaler_path (str): Path to the scaler.

    Returns:
        pd.DataFrame: DataFrame with standardized continuous variables.
    """
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)

    # Save the scaler
    joblib.dump(value=scaler, filename=scaler_path)

    # Transform the continuous variables
    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
    return cont_vars


def combine_cat_and_cont(cont_vars: pd.DataFrame, cat_vars: pd.DataFrame) -> pd.DataFrame:
    """Combine continuous and categorical variables into a single DataFrame.

    Args:
        cont_vars (pd.DataFrame): DataFrame containing continuous variables.
        cat_vars (pd.DataFrame): DataFrame containing categorical variables.

    Returns:
        pd.DataFrame: Combined DataFrame.
    """

    # Reset indices to keep row alignment
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)

    data = pd.concat([cont_vars, cat_vars], axis=1)
    return data


def save_data_drift(data: pd.DataFrame) -> None:
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
    with open("./artifacts/columns_drift.json", "w+") as f:
        json.dump(data_columns, f)

    # Save full training data
    data.to_csv("./artifacts/training_data.csv", index=False)


def bin_source_column(data: pd.DataFrame) -> pd.DataFrame:
    """Bin the 'source' column into broader categories.

    Args:
        data (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with binned 'source' column.
    """
    data["bin_source"] = data["source"]
    values_list = ["li", "organic", "signup", "fb"]
    data.loc[~data["source"].isin(values_list), "bin_source"] = "Others"
    mapping = {"li": "socials", "fb": "socials", "organic": "group1", "signup": "group1"}

    data["bin_source"] = data["source"].map(mapping)

    return data


def save_gold_medallion(data: pd.DataFrame) -> None:
    """Save the gold medallion schema and dataset artifacts.

    Args:
        data (pd.DataFrame): DataFrame containing the data (cont + cat)
    """
    data.to_csv("./artifacts/train_data_gold.csv", index=False)


if __name__ == "__main__":
    print("Starting preprocessing...\n")
    create_directories()
    data = load_data("./artifacts/raw_data.csv")
    time_limit_data(data)
    data = feature_selection(data)
    data = data_cleaning(data)
    data = create_cat_cols(data)
    cont_vars, cat_vars = separate_cat_and_cont_cols(data)
    cont_vars = outliers(cont_vars, z=2.0)
    cont_vars = impute_continuous(cont_vars)
    cat_vars = impute_categorical(cat_vars)
    cont_vars = standardize_continuous(cont_vars)
    data = combine_cat_and_cont(cont_vars, cat_vars)
    save_data_drift(data)
    data = bin_source_column(data)
    save_gold_medallion(data)
    print("Preprocessing completed and artifacts saved.")
