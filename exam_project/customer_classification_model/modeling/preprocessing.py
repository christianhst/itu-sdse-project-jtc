from datetime import datetime
from  customer_classification_model.data_utils import impute_missing_values

import json

import numpy as np
import pandas as pd


# Date limits for data
max_date = "2024-01-31"
min_date = "2024-01-01"


def time_limit_data(data: pd.DataFrame, max_date=max_date, min_date=min_date):
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
    result = data.lead_indicator.value_counts(normalize=True)

    print("Target value counter")
    for val, n in zip(result.index, result):
        print(val, ": ", n)
    return data

def create_cat_cols(data: pd.DataFrame) -> pd.DataFrame:
    """Convert specified columns to categorical (object) type.

    Args:
        data (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with categorical columns converted to object type.
    """
    vars = [
    "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
    ]

    for col in vars:
        data[col] = data[col].astype("object")
        print(f"Changed {col} to object type")
    return data

def separate_cat_and_cont_cols(data: pd.DataFrame) -> pd.DataFrame:
    """xx
    
    Args:
        data (pd.DataFrame): Input DataFrame.
    
    Returns:
    """

    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]

    print("\nContinuous columns: \n")
    print(list(cont_vars.columns), indent=4)
    print("\n Categorical columns: \n")
    print(list(cat_vars.columns), indent=4)
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
        lambda x: x.clip(
            lower=x.mean() - z * x.std(),
            upper=x.mean() + z * x.std()
        )
    )
    
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
    return cat_vars