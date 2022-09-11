from numbers import Real
import logging
from typing import Any, Dict, List

import category_encoders as ce
import lightgbm as lgb
import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import log_loss, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from model.helpers import lgb_f1_score
from settings import LGBM_PARAMS, EXCLUDE_COLUMNS, GRID_SEARCH_PARAMS

logger = logging.getLogger(__name__)


class Estimator(object):

    def __init__(self,
                 category_encoder: Any = None,
                 lgb_model: lgb.LGBMModel = None,
                 lgbm_params: Dict = LGBM_PARAMS) -> None:
        if lgb_model is None:
            lgb_model = lgb.LGBMClassifier(**lgbm_params)
        if category_encoder is None:
            category_encoder = ce.CatBoostEncoder()
        self.lgb_model = lgb_model
        self.category_encoder = category_encoder

    @staticmethod
    def _preprocess_data(X: pd.DataFrame,
                         exclude_columns: List = EXCLUDE_COLUMNS,
                         fill_value: Real = np.nan) -> pd.DataFrame:
        columns = [col for col in X.columns if col not in exclude_columns]
        X = X.loc[:, columns]
        X = X.fillna(fill_value)
        return X

    def _fit_encoder(self,
                     X: pd.DataFrame,
                     *args,
                     encoder_params: Dict = None,
                     **kwargs) -> None:
        X = self._preprocess_data(X)
        categorical_columns = []
        for column_name in X:
            dtype = X[column_name].dtype
            if dtype == "object" or dtype.name == "category":
                categorical_columns.append(column_name)
        if encoder_params is None:
            encoder_params = {'random_state': 42}
        encoder_params = {**encoder_params, "cols": categorical_columns}
        self.category_encoder.set_params(**encoder_params)
        self.category_encoder.fit(X, *args, **kwargs)

    def _get_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._preprocess_data(X)
        return self.category_encoder.transform(X)

    def fit(self,
            X: pd.DataFrame,
            y: np.ndarray,
            *args,
            encoder_params: Dict = None,
            **kwargs) -> None:
        self._fit_encoder(X, y, encoder_params=encoder_params)
        X_transformed = self._get_features(X)
        if eval_set := kwargs.get("eval_set"):
            kwargs["eval_set"] = [(self._get_features(X), y) for X, y in eval_set]
        self.lgb_model.fit(X_transformed, y, *args, **kwargs)

    def predict(self, X: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        X_transformed = self._get_features(X)
        return self.lgb_model.predict(X_transformed, *args, **kwargs)

    def predict_proba(self, X: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        X_transformed = self._get_features(X)
        return self.lgb_model.predict_proba(X_transformed, *args, **kwargs)

    def evaluate(self,
                 X: pd.DataFrame,
                 y: np.ndarray,
                 *args,
                 labels: List = None,
                 **kwargs) -> Dict:
        if labels is None:
            # assume binary problem
            labels = [0, 1]
        y_pred = self.predict(X, *args, **kwargs)
        pred_proba = self.predict_proba(X, *args, **kwargs)
        loss = log_loss(y, pred_proba, labels=labels)
        f1 = f1_score(y, y_pred)
        return {"f1_score": f1,
                "log_loss": loss}

    def tune_parameters(self,
                        X: pd.DataFrame,
                        y: np.ndarray,
                        cross_validator: Any = None,
                        metric: str = "f1_score",
                        encoder_params: Dict = None,
                        **kwargs) -> Dict:
        if cross_validator is None:
            if "random_state" in kwargs and "shuffle" not in "kwargs":
                kwargs.update(shuffle=True)
            cross_validator = StratifiedKFold(**kwargs)
        best_metrics = {"f1_score": -np.inf,
                        "log_loss": -np.inf}
        best_params = {}
        for params in ParameterGrid(GRID_SEARCH_PARAMS):
            with mlflow.start_run():
                self.lgb_model.set_params(**params)
                fold_metrics = []
                mlflow.log_params(params)
                for train_index, test_index in cross_validator.split(X, y):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    self.fit(X_train, y_train,
                             encoder_params=encoder_params,
                             eval_set=[(X_test, y_test)],
                             eval_metric=lgb_f1_score,
                             verbose=-1)
                    metrics = self.evaluate(X_test, y_test)
                    fold_metrics.append(metrics)
                avg_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
                for metric, value in avg_metrics.items():
                    mlflow.log_metric(metric, value)
                mlflow.lightgbm.log_model(self.lgb_model, artifact_path="models_mlflow")
                if avg_metrics[metric] > best_metrics[metric]:
                    best_metrics = avg_metrics
                    best_params = self.lgb_model.get_params(deep=False)
        return best_params