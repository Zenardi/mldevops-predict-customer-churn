"""
Unit tests and logging for churn_library.py functions.

Validates each step of the customer churn ML pipeline and logs results
(INFO for success, ERROR for failure) to ./logs/churn_library.log.

Author: Student
Date: 2024
"""

import logging
import os

import churn_library as cls

LOGS_DIR = "./logs"
LOG_FILE = os.path.join(LOGS_DIR, "churn_library.log")
DATA_PATH = "./data/bank_data.csv"

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def test_import(import_data):
    """Test data import."""
    try:
        df = import_data(DATA_PATH)
        logging.info("Testing import_data: SUCCESS - file loaded")
    except FileNotFoundError as err:
        logging.error("Testing import_data: ERROR - file not found at %s", DATA_PATH)
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info(
            "Testing import_data: SUCCESS - dataframe shape %s", df.shape
        )
    except AssertionError as err:
        logging.error(
            "Testing import_data: ERROR - dataframe has unexpected shape %s",
            df.shape,
        )
        raise err


def test_eda(perform_eda):
    """Test EDA."""
    try:
        df = cls.import_data(DATA_PATH)
        perform_eda(df)

        expected_files = [
            'churn_distribution.png',
            'customer_age_distribution.png',
            'marital_status_distribution.png',
            'total_trans_ct_distribution.png',
            'heatmap.png',
        ]
        for fname in expected_files:
            fpath = os.path.join(cls.EDA_DIR, fname)
            assert os.path.isfile(fpath), f"Missing EDA file: {fpath}"

        logging.info("Testing perform_eda: SUCCESS - all EDA images saved")
    except Exception as err:
        logging.error("Testing perform_eda: ERROR - %s", str(err))
        raise err


def test_encoder_helper(encoder_helper):
    """Test encoding."""
    try:
        df = cls.import_data(DATA_PATH)
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

        category_lst = [
            'Gender', 'Education_Level', 'Marital_Status',
            'Income_Category', 'Card_Category',
        ]
        df = encoder_helper(df, category_lst, 'Churn')

        for col in category_lst:
            encoded_col = f'{col}_Churn'
            assert encoded_col in df.columns, f"Missing encoded column: {encoded_col}"

        logging.info(
            "Testing encoder_helper: SUCCESS - encoded columns present"
        )
    except Exception as err:
        logging.error("Testing encoder_helper: ERROR - %s", str(err))
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """Test feature engineering."""
    try:
        df = cls.import_data(DATA_PATH)
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        category_lst = [
            'Gender', 'Education_Level', 'Marital_Status',
            'Income_Category', 'Card_Category',
        ]
        df = cls.encoder_helper(df, category_lst, 'Churn')

        x_train, x_test, y_train, y_test = perform_feature_engineering(df, 'Churn')

        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0

        logging.info(
            "Testing perform_feature_engineering: SUCCESS - "
            "train/test split shapes: x_train=%s x_test=%s",
            x_train.shape, x_test.shape,
        )
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: ERROR - %s", str(err)
        )
        raise err


def test_train_models(train_models):
    """Test model training."""
    try:
        df = cls.import_data(DATA_PATH)
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        category_lst = [
            'Gender', 'Education_Level', 'Marital_Status',
            'Income_Category', 'Card_Category',
        ]
        df = cls.encoder_helper(df, category_lst, 'Churn')
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df, 'Churn')

        train_models(x_train, x_test, y_train, y_test)

        expected_models = ['rfc_model.pkl', 'logistic_model.pkl']
        for fname in expected_models:
            fpath = os.path.join(cls.MODELS_DIR, fname)
            assert os.path.isfile(fpath), f"Missing model file: {fpath}"

        expected_images = [
            'rf_results.png',
            'logistic_results.png',
            'feature_importances.png',
            'roc_curve_result.png',
            'precision_recall_curves.png',
            'confusion_matrices.png',
        ]
        for fname in expected_images:
            fpath = os.path.join(cls.RESULTS_DIR, fname)
            assert os.path.isfile(fpath), f"Missing results image: {fpath}"

        logging.info("Testing train_models: SUCCESS - models and images saved")
    except Exception as err:
        logging.error("Testing train_models: ERROR - %s", str(err))
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

    print("Tests completed. Check logs for details.")
