import logging
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from sklearn.model_selection import train_test_split

from model.estimator import Estimator
from model.helpers import lgb_f1_score
from settings import DATA_PATH


EXPERIMENT_NAME = "loan-prediction"

logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def main_flow(data_path=DATA_PATH):
    model = Estimator()
    data = pd.read_csv(data_path)
    X = data.loc[:, data.columns != "is_bad"]
    y = data["is_bad"].values
    best_params = model.tune_parameters(X, y, random_state=42)
    train_best_model(model, X, y, best_params)
    register_model()


def train_best_model(model: Estimator, X, y, best_params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with mlflow.start_run():
        model.lgb_model.set_params(**best_params)
        mlflow.log_params(best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=lgb_f1_score,
            verbose=-1
        )
        mlflow.lightgbm.log_model(model.lgb_model, artifact_path="models_mlflow")


def register_model(test_metric='log_loss', registered_model_name='loan-predictor'):

    client = MlflowClient()
    # select the model with the lowest test log loss
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=[f"metrics.{test_metric} ASC"],
        max_results=1)[0]
    logger.info(f"Best test log loss {best_run.data.metrics[test_metric]}")
    # Register the best model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )


if __name__ == "__main__":
    main_flow()
