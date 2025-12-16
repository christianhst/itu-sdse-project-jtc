# Customer Classification model

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Exam project in the course Data Science in Production: MLOps and Software Engineering (Autumn 2025). This project builds a model that identifies users on the website that are new possible customers. This is done by collecting behaviour data from the users as input, and the target is whether they converted/turned into customers - essentially a classification problem. 

The Cookiecutter Data Science template (ccds) has been used for the project structure and configured accordingly. The functioning code consists of python scripts with different purposes in the MLOps cycle and has been wrapped in a dagger pipeline written in Go. This is then orchestrated in a GitHub workflow. This ensures consistent behavior across different environments since it is run in a containerized structure.

The structure of the repository/project is as follows:

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- This README describing the structure of the project and how to run the code.
├── data
│   ├── external       <- Data from third party sources (not used in this project).
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data set for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project
│
├── models             <- Trained models and model summaries
│
├── notebooks          <- Jupyter notebooks. This contains the initial, provided notebook which has been 
│                         refactored (main.ipynb)
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         customer_classification_model
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials (not used).
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc. (not used)
│   └── figures        <- Generated graphics and figures to be used in reporting (not used)
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
│
├── ci  
│    ├── pipeline.go   <- Dagger pipeline written in Go to run the project   
│    ├── go.sum        <- Go dependency checksums for reproducible builds
│    └── go.mod        <- Go module definition for the pipeline
│
└── customer_classification_model   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes customer_classification_model a Python module
    │
    ├── constants.py            <- Constants used in the project
    │
    ├── data_utils.py           <- Utility functions for operation on data
    │
    ├── mlflow_utils.py         <- utility functions related to MLFlow
    │
    ├── modeling                
        ├── __init__.py 
        ├── preprocessing.py    <- Code to preprocess data        
        ├── train.py            <- Code to train and evaluate models
        ├── model_selection.py  <- Code to select best model 
        └── deploy.py           <- Code to deploy best model 

```

--------

## Setup and how to run the code and generate the model artifact

1. Create and activate a virtual environment:
   - macOS/Linux:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Docker Desktop (or another Docker engine) and the Dagger CLI for pipeline execution.
4. Install Go 1.25+ if you want to run the pipeline locally.

## Running locally

### Individual scripts

1. Fetch the latest raw dataset tracked by DVC:
   ```bash
   cd exam_project/data/raw
   dvc update raw_data.csv.dvc
   ```
2. Execute the preprocessing, training, model selection, and deployment scripts:
   ```bash
   cd ../../customer_classification_model
   python modeling/preprocessing.py
   python modeling/train.py
   python modeling/model_selection.py
   python modeling/deploy.py
   ```

### Dagger pipeline

1. Ensure Docker is running.
2. Run the pipeline, which installs dependencies, executes the four scripts, and exports artifacts to `exam_project/ci/output`:
   ```bash
   cd exam_project/ci
   go run .
   ```

### GitHub Actions

In GitHub, open the **Actions** tab, select **Customer Classification Model Pipeline (train, upload and test model)**, and click **Run workflow**. The workflow runs the same Dagger pipeline and generates the trained model artifact.

## Data

All data is placed in `exam_project/data` and follows the CCDS structure (`raw → interim → processed`). The raw dataset is versioned with DVC and stays out of Git history.

- `raw/raw_data.csv.dvc` is a pointer file that knows how to download the CSV from [`Jeppe-T-K/itu-sdse-project-data`](https://raw.githubusercontent.com/Jeppe-T-K/itu-sdse-project-data/refs/heads/main/raw_data.csv).
- `raw/raw_data.csv` consist of 12,346 rows of website leads collected between `2024-01-01` and `2024-01-31`. Columns include:
  - Target: `lead_indicator` (`1` = the visitor converted, `0` = no conversion).
  - Identifiers and lifecycle timestamps: `lead_id`, `customer_code`, `date_part`, `first_booking`, `last_seen`.
  - Behavior features: `n_visits`, `time_spent`, `purchases`, `visited_learn_more_before_booking`, `visited_faq`.
  - Channel metadata: `source`, `domain`, `country`, `customer_group`, `onboarding`, `marketing_consent`, `existing_customer`.

Fetch or refresh the raw file with:

```bash
cd exam_project/data/raw
dvc update raw_data.csv.dvc   # downloads raw_data.csv next to the .dvc file
```
