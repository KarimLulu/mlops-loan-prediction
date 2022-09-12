import logging

import numpy as np
import mlflow
import pandas as pd
import lightgbm as lgb
from settings import LGBM_PARAMS, GRID_SEARCH_PARAMS
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.feature_extraction import DictVectorizer

logger = logging.getLogger(__name__)


def read_dataframe(filename: str):
    df = pd.read_csv(filename)
    columns = ['emp_length', 'home_ownership', 'annual_inc', 'is_bad']
    df = df[columns]
    df['home_ownership'] = df['home_ownership'].astype(str)
    return df


def add_features(df: pd.DataFrame):
    categorical = ['home_ownership']
    numerical = ['emp_length', 'annual_inc']
    X, y = df.loc[:, categorical + numerical], df['is_bad']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_dicts = X_train.to_dict(orient='records')
    test_dicts = X_test.to_dict(orient='records')
    return train_dicts, test_dicts, y_train, y_test


def evaluate_model(model, y_test, test_features):
    pred_proba = model.predict_proba(test_features)
    loss = log_loss(y_test, pred_proba, labels=y_test)
    return loss


def tune_parameters(path):
    df = read_dataframe(path)
    train_dicts, test_dicts, y_train, y_test = add_features(df)
    pipeline = make_pipeline(DictVectorizer(), lgb.LGBMClassifier(**LGBM_PARAMS))
    best_loss = -np.inf
    best_params = {}
    for params in ParameterGrid(GRID_SEARCH_PARAMS):
        with mlflow.start_run():
            pipeline.set_params(**params)
            mlflow.log_params(params)
            pipeline.fit(train_dicts, y_train)
            loss = evaluate_model(pipeline, y_test, test_dicts)
            mlflow.log_metric('log_loss', loss)
            mlflow.sklearn.log_model(pipeline, artifact_path="prediction_service")
            if loss > best_loss:
                best_loss = loss
                best_params = pipeline.get_params()
    return best_params, pipeline
