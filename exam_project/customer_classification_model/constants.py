import datetime

# Constants used:
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "./artifacts/train_data_gold.csv"
data_version = "00000"
experiment_name = current_date
artifact_path = "model"
model_name = "lead_model"
