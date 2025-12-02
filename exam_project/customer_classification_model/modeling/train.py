import json
import warnings

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRFClassifier

from customer_classification_model.constants import data_gold_path, experiment_name
from customer_classification_model.data_utils import create_dummy_cols

warnings.filterwarnings("ignore")


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


def data_split_train_test(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The training and testing sets for features and labels.
    """
    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.15, stratify=y
    )
    return X_train, X_test, y_train, y_test


def xgboost_fit(X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    """
    Fit an XGBoost model using randomized search cross-validation.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        RandomizedSearchCV: The fitted randomized search model.
    """

    model = XGBRFClassifier(random_state=42)
    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"],
    }

    model_grid = RandomizedSearchCV(
        model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10
    )

    model_grid.fit(X_train, y_train)
    return model_grid


def xgboost_model_evaluation(
    model_grid: RandomizedSearchCV,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    #placeholder
    """

    best_model_xgboost_params = model_grid.best_params_
    print("Best xgboost params")
    print(best_model_xgboost_params)

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)
    print("Accuracy train", accuracy_score(y_pred_train, y_train))
    print("Accuracy test", accuracy_score(y_pred_test, y_test))

    conf_matrix = confusion_matrix(y_test, y_pred_test)
    print("Test actual/predicted\n")
    print(
        pd.crosstab(
            y_test, y_pred_test, rownames=["Actual"], colnames=["Predicted"], margins=True
        ),
        "\n",
    )
    print("Classification report\n")
    print(classification_report(y_test, y_pred_test), "\n")

    conf_matrix = confusion_matrix(y_train, y_pred_train)
    print(conf_matrix)
    print("Train actual/predicted\n")
    print(
        pd.crosstab(
            y_train, y_pred_train, rownames=["Actual"], colnames=["Predicted"], margins=True
        ),
        "\n",
    )
    print("Classification report\n")
    print(classification_report(y_train, y_pred_train), "\n")


def xgboost_save_best_model(
    model_grid: RandomizedSearchCV, y_train: pd.Series, y_pred_train: pd.Series
) -> tuple[str, dict]:
    """
    # placeholder
    """
    xgboost_model = model_grid.best_estimator_
    xgboost_model_path = "./artifacts/lead_model_xgboost.json"
    xgboost_model.save_model(xgboost_model_path)

    xgb_report = classification_report(y_train, y_pred_train, output_dict=True)
    print(xgb_report)
    return xgboost_model_path, xgb_report


class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


def build_lr_search() -> RandomizedSearchCV:
    """
    Create a RandomizedSearchCV for LogisticRegression.

    Args:
        None

    Returns:
        RandomizedSearchCV: The randomized search object
    """

    model = LogisticRegression()

    params = {
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "penalty": ["none", "l1", "l2", "elasticnet"],
        "C": [100, 10, 1.0, 0.1, 0.01],
    }

    model_grid = RandomizedSearchCV(
        estimator=model, param_distributions=params, n_iter=10, cv=3, verbose=3
    )
    return model_grid


def train_and_eval_lr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[LogisticRegression, dict, dict, str]:
    """
    Fit LR with randomized search and return best model + metrics.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_test (pd.DataFrame): Testing features
        y_test (pd.Series): Testing labels

    Returns:
        tuple[LogisticRegression, dict, dict]: Best model, classification report, best params
    """

    with mlflow.start_run(run_name="logistic_regression"):
        model_grid = build_lr_search()
        model_grid.fit(X_train, y_train)

        best_model = model_grid.best_estimator_

        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        # log artifacts
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_test))
        mlflow.log_artifacts("artifacts", artifact_path="model")
        mlflow.log_param("data_version", "00000")

        # store model for model interpretability
        lr_model_path = "./artifacts/lead_model_lr.pkl"
        joblib.dump(value=best_model, filename=lr_model_path)

        # Custom python model for predicting probability
        mlflow.pyfunc.log_model(
            "model", python_model=lr_wrapper(best_model)
        )  # I changed original model to best_model, for me it makes more sense

    model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)
    best_model_lr_params = model_grid.best_params_

    print("Best lr params")
    print(best_model_lr_params)
    print("Accuracy train:", accuracy_score(y_train, y_pred_train))
    print("Accuracy test:", accuracy_score(y_test, y_pred_test))

    # conf_matrix_test = confusion_matrix(y_test, y_pred_test) NOT ACCESSED
    print("Test actual/predicted\n")

    print(
        pd.crosstab(
            y_test, y_pred_test, rownames=["Actual"], colnames=["Predicted"], margins=True
        ),
        "\n",
    )
    print("Classification report\n")
    print(classification_report(y_test, y_pred_test), "\n")

    # conf_matrix_train = confusion_matrix(y_train, y_pred_train) NOT ACCESSED
    print("Train actual/predicted\n")

    print(
        pd.crosstab(
            y_train, y_pred_train, rownames=["Actual"], colnames=["Predicted"], margins=True
        ),
        "\n",
    )
    print("Classification report\n")
    print(classification_report(y_train, y_pred_train), "\n")

    # model_results = { lr_model_path: model_classification_report}
    # # NOT ACCESSED, this line does the same as 273 line - model_classification_report
    print(model_classification_report["weighted avg"]["f1-score"])

    return best_model, model_classification_report, best_model_lr_params, lr_model_path


def save_columns_and_model_results(X_train: pd.DataFrame, model_results: dict) -> None:
    """
    Save column list and model results to artifacts.

    Args:
        X_train (pd.DataFrame): Training features
        model_results (dict): Model results dictionary

    Returns:
        None
    """

    # Save column list
    column_list_path = "./artifacts/columns_list.json"
    with open(column_list_path, "w+") as columns_file:
        columns = {"column_names": list(X_train.columns)}
        print(columns)
        json.dump(columns, columns_file)
    print("Saved column list to ", column_list_path)

    # Save model results
    model_results_path = "./artifacts/model_results.json"
    with open(model_results_path, "w+") as results_file:
        json.dump(model_results, results_file)
    print("Saved model_results to ", model_results_path)


if __name__ == "__main__":
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    data = load_train_data(data_gold_path)
    cat_vars, other_vars = data_type_split(data)
    data = one_hot_cat_cols(cat_vars, other_vars)
    X_train, X_test, y_train, y_test = data_split_train_test(data)

    # Train and evaluate XGBoost model
    model_grid = xgboost_fit(X_train, y_train)
    xgboost_model_evaluation(model_grid, X_train, X_test, y_train, y_test)
    xgb_model_path, xgb_report = xgboost_save_best_model(
        model_grid, y_train, model_grid.predict(X_train)
    )

    # Train and evaluate Logistic Regression model
    best_model_lr, model_classification_report, best_model_lr_params, lr_model_path = (
        train_and_eval_lr(X_train, y_train, X_test, y_test)
    )

    # Combine model_results from BOTH models
    model_results = {
        lr_model_path: model_classification_report,
        xgb_model_path: xgb_report,
    }

    # Save columns + model_results once
    save_columns_and_model_results(X_train, model_results)
