# Customer Classification model

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Exam project in the course Data Science in Production: MLOps and Software Engineering (Autumn 2025). This project builds a model that identifies users on the website that are new possible customers. This is done by collecting behaviour data from the users as input, and the target is whether they converted/turned into customers - essentially a classification problem. 

The Cookiecutter Data Science template (ccds) has been used for the project structure and configured accordingly. The functioning code consists of python scripts with different purposes in the MLOps cycle and has been wrapped in a dagger pipeline written in Go. This is then run through a GitHub workflow. This ensures consistent behavior across different environments since it is run in a conntainerized structure.

The structure of the repository/project is as follows:

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources (not used in this project).
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data set for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained models and model summaries
│
├── notebooks          <- Jupyter notebooks. This contains the initial notebook which has been 
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
│    ├── pipeline.go   <- Dagger pipeline written in go to run the project   
│    ├── go.sum        <- x
│    └── go.mod        <- x
│        
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

## Setup

To run the scripts manually on your local machine, you need to follow these steps before: 
- create virtual environment
- activate virtual environmnet
- download dependcies using requirements.txt file

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



To run project inside the github go to the actions pain, select "Customer Classificationm Model Pipeline (train, upload and test model)" an press "Run workflow" buttom. 