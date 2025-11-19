import os

import pandas as pd


def create_directories():
    """Create local directories used for artifacts and MLflow tracking."""
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("mlruns/.trash", exist_ok=True)
    print("Created artifact and mlruns directories")


def describe_numeric_col(x):
    """Compute basic descriptive statistics for a numeric series.

    Args:
        x (pd.Series): Numeric column to summarize.

    Returns:
        pd.Series: Series containing count, missing, mean, min, and max values.
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"],
    )


def impute_missing_values(x, method="mean"):
    """Fill missing values in a series using a numerical or categorical strategy.

    Args:
        x (pd.Series): Column whose missing values will be imputed.
        method (str, optional): Strategy for numerical columns, either ``"mean"``
            or ``"median"``. Defaults to ``"mean"``.

    Returns:
        pd.Series: Series with missing values imputed.
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method == "mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x


def create_dummy_cols(df, col):
    """Encode a categorical column using one-hot encoding and drop the original.

    Args:
        df (pd.DataFrame): DataFrame containing the column to encode.
        col (str): Name of the categorical column to convert into dummy columns.

    Returns:
        pd.DataFrame: DataFrame with dummy columns appended and the original
            categorical column removed.
    """
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df
