# TODO:
# Add a module-level docstring describing:
# - Purpose of this file
# - Author
# - Date created

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

# TODO: add required imports
# Example:
# import joblib
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, RocCurveDisplay
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

EDA_DIR = "./images/eda"
RESULTS_DIR = "./images/results"
MODELS_DIR = "./models"
DATA_PATH = "./data/bank_data.csv"


def create_output_directories():
    """
    Create output directories used by the project.

    input:
            None
    output:
            None
    """
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def import_data(pth):
    """
    Return a dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    # TODO: implement
    # Hint:
    # df = pd.read_csv(pth)
    # return df
    pass


def perform_eda(df):
    """
    Perform EDA on df and save figures.

    input:
            df: pandas dataframe
    output:
            None
    """
    create_output_directories()

    # TODO: implement
    # Suggested steps:
    # 1. Create a binary churn column if needed
    # 2. Plot key distributions
    # 3. Plot a correlation heatmap
    # 4. Save figures into EDA_DIR
    pass


def encoder_helper(df, category_lst, response):
    """
    Encode categorical features.

    input:
            df: pandas dataframe
            category_lst: list of categorical columns
            response: response column name
    output:
            df: updated dataframe
    """
    # TODO: implement
    pass


def perform_feature_engineering(df, response):
    """
    Split dataset into train and test sets.

    input:
              df: pandas dataframe
              response: response column name
    output:
              x_train, x_test, y_train, y_test
    """
    # TODO: implement
    pass


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    Save classification reports as images.

    input:
            predictions and labels
    output:
            None
    """
    create_output_directories()

    # TODO: implement
    pass


def feature_importance_plot(model, x_data, output_pth):
    """
    Save feature importance plot.

    input:
            model, x_data, output path
    output:
            None
    """
    # TODO: implement
    pass


def train_models(x_train, x_test, y_train, y_test):
    """
    Train models and save outputs.

    input:
            train/test data
    output:
            None
    """
    create_output_directories()

    # TODO: implement
    pass


if __name__ == "__main__":
    create_output_directories()

    df = import_data(DATA_PATH)

    perform_eda(df)

    category_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]

    df = encoder_helper(df, category_columns, "Churn")
    x_train, x_test, y_train, y_test = perform_feature_engineering(df, "Churn")
    train_models(x_train, x_test, y_train, y_test)
