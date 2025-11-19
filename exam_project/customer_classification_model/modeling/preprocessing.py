# import warnings

from datetime import datetime
import json

import pandas as pd

# warnings.filterwarnings("ignore")
# pd.set_option("display.float_format", lambda x: "%.3f" % x)

# Date limits for data
max_date = "2024-01-31"
min_date = "2024-01-01"


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path)

    print("Total rows:", data.count())
    print(data.head().to_string())
    return data


def time_limit_data(data: pd.DataFrame, max_date=max_date, min_date=min_date) -> pd.DataFrame:
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
