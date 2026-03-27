# TODO:
# Add a module-level docstring describing:
# - Purpose of this file
# - Author
# - Date created

import logging
import os

import churn_library as cls

LOGS_DIR = "./logs"
LOG_FILE = os.path.join(LOGS_DIR, "churn_library.log")
DATA_PATH = "./data/bank_data.csv"

# TODO:
# configure logging to write INFO and ERROR messages
# to a .log file inside the ./logs directory


def test_import(import_data):
    """Test data import."""
    try:
        df = import_data(DATA_PATH)

        # TODO: add logging for success

    except FileNotFoundError as err:

        # TODO: add logging for file not found

        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:

        # TODO: add logging for failure

        raise err


def test_eda(perform_eda):
    """Test EDA."""
    try:
        df = cls.import_data(DATA_PATH)
        perform_eda(df)

        # TODO:
        # assert output files exist

        # TODO: logging success

    except Exception as err:

        # TODO: logging failure

        raise err


def test_encoder_helper(encoder_helper):
    """Test encoding."""
    try:
        df = cls.import_data(DATA_PATH)

        # TODO:
        # create response column
        # call encoder_helper
        # assert encoded columns exist

        # TODO: logging success

    except Exception as err:

        # TODO: logging failure

        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """Test feature engineering."""
    try:
        df = cls.import_data(DATA_PATH)

        # TODO:
        # prepare data
        # call function
        # assert outputs

        # TODO: logging success

    except Exception as err:

        # TODO: logging failure

        raise err


def test_train_models(train_models):
    """Test model training."""
    try:
        df = cls.import_data(DATA_PATH)

        # TODO:
        # prepare data
        # call train_models
        # assert model files + images exist

        # TODO: logging success

    except Exception as err:

        # TODO: logging failure

        raise err


if __name__ == "__main__":
    # TODO: ensure logs directory exists

    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

    print("Tests completed. Check logs for details.")
