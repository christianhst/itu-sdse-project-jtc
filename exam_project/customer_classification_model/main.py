# Only for deployment purposes

from data_utils import create_directories
from dataset import load_data
from modeling.preprocessing import time_limit_data

if __name__ == "__main__":
    create_directories()
    data = load_data("./artifacts/raw_data.csv")
    data = time_limit_data(data)
