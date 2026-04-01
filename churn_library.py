"""
Library of functions to identify credit card customers likely to churn.

Implements the full ML pipeline: data loading, EDA, feature engineering,
model training, evaluation, and result persistence.

Author: Student
Date: 2024
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ["QT_QPA_PLATFORM"] = "offscreen"

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
    return pd.read_csv(pth)


def perform_eda(df):
    """
    Perform EDA on df and save figures.

    input:
            df: pandas dataframe
    output:
            None
    """
    create_output_directories()

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(os.path.join(EDA_DIR, 'churn_distribution.png'))
    plt.close()

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(os.path.join(EDA_DIR, 'customer_age_distribution.png'))
    plt.close()

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join(EDA_DIR, 'marital_status_distribution.png'))
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join(EDA_DIR, 'total_trans_ct_distribution.png'))
    plt.close()

    plt.figure(figsize=(20, 10))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(EDA_DIR, 'heatmap.png'))
    plt.close()


def encoder_helper(df, category_lst, response):
    """
    Encode categorical features using mean response encoding.

    input:
            df: pandas dataframe
            category_lst: list of categorical columns
            response: response column name
    output:
            df: updated dataframe with new encoded columns
    """
    for category in category_lst:
        df[f'{category}_{response}'] = df[category].map(
            df.groupby(category)[response].mean()
        )
    return df


def perform_feature_engineering(df, response):
    """
    Split dataset into train and test sets.

    input:
              df: pandas dataframe
              response: response column name
    output:
              x_train, x_test, y_train, y_test
    """
    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn',
    ]

    x_data = df[keep_cols]
    y_data = df[response]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42
    )
    return x_train, x_test, y_train, y_test


def classification_report_image(  # pylint: disable=too-many-arguments,too-many-positional-arguments
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
            y_train: training response values
            y_test: test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    output:
            None
    """
    create_output_directories()

    plt.rc('figure', figsize=(7, 5))

    plt.figure()
    plt.text(0.01, 1.25, 'Random Forest Train', fontsize=10,
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             fontsize=10, fontproperties='monospace')
    plt.text(0.01, 0.6, 'Random Forest Test', fontsize=10,
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             fontsize=10, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_DIR, 'rf_results.png'), bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.text(0.01, 1.25, 'Logistic Regression Train', fontsize=10,
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_lr)),
             fontsize=10, fontproperties='monospace')
    plt.text(0.01, 0.6, 'Logistic Regression Test', fontsize=10,
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_lr)),
             fontsize=10, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_DIR, 'logistic_results.png'),
                bbox_inches='tight')
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    """
    Save feature importance plot.

    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
            None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    """
    Train models and save outputs.

    input:
            x_train: X training data
            x_test: X testing data
            y_train: y training data
            y_test: y testing data
    output:
            None
    """
    create_output_directories()

    rfc = RandomForestClassifier(random_state=42)
    lrc = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=3000))
    ])

    param_dist = {
        'n_estimators': [200, 300],
        'max_features': ['sqrt'],
        'max_depth': [5, 8, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'criterion': ['gini'],
    }

    cv_rfc = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=param_dist,
        n_iter=12,
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image(
        y_train, y_test,
        y_train_preds_lr, y_train_preds_rf,
        y_test_preds_lr, y_test_preds_rf,
    )

    feature_importance_plot(
        cv_rfc.best_estimator_,
        x_test,
        os.path.join(RESULTS_DIR, 'feature_importances.png'),
    )

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    lrc_plot = RocCurveDisplay.from_estimator(lrc, x_test, y_test)
    RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, x_test, y_test, ax=ax, alpha=0.8
    )
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve_result.png'))
    plt.close()

    joblib.dump(cv_rfc.best_estimator_, os.path.join(MODELS_DIR, 'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(MODELS_DIR, 'logistic_model.pkl'))


if __name__ == "__main__":
    create_output_directories()

    data_df = import_data(DATA_PATH)

    perform_eda(data_df)

    category_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]

    data_df = encoder_helper(data_df, category_columns, "Churn")
    train_x, test_x, train_y, test_y = perform_feature_engineering(data_df, "Churn")
    train_models(train_x, test_x, train_y, test_y)
