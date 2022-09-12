from datetime import timedelta
import os
import logging

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from prefect import flow, task
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.infrastructure import Process
from prefect.task_runners import SequentialTaskRunner
from sklearn.model_selection import train_test_split

from prediction_service.estimator import read_dataframe, add_features, tune_parameters, evaluate_model
from settings import DATA_PATH, EXPERIMENT_NAME, TRACKING_URI


logger = logging.getLogger(__name__)
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


#@task
def train_best_model(model, data_path, best_params):
    df = read_dataframe(data_path)
    train_dicts, test_dicts, y_train, y_test = add_features(df)
    with mlflow.start_run():
        model.set_params(**best_params)
        mlflow.log_params(best_params)
        model.fit(
            train_dicts, y_train,
        )
        loss = evaluate_model(model, y_test, test_dicts)
        mlflow.log_metric('log_loss', loss)
        mlflow.sklearn.log_model(model, artifact_path="prediction_service")


#@task
def register_model(test_metric='log_loss', registered_model_name='loan-predictor'):

    client = MlflowClient()
    # select the prediction_service with the lowest test log loss
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=[f"metrics.{test_metric} ASC"],
        max_results=1)[0]
    logger.info(f"Best test log loss {best_run.data.metrics[test_metric]}")
    # Register the best prediction_service
    model_uri = f"runs:/{best_run.info.run_id}/prediction_service"
    model_version = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )


#@flow(task_runner=SequentialTaskRunner())
def main_flow(data_path=DATA_PATH):
    registered_model_name = os.getenv('MLFLOW_MODEL_NAME', 'loan-predictor')
    best_params, model = tune_parameters(data_path)
    train_best_model(model, data_path, best_params)
    register_model(registered_model_name=registered_model_name)


# deployment = Deployment.build_from_flow(
#     flow=main_flow,
#     name="model_training",
#     schedule=IntervalSchedule(interval=timedelta(weeks=1)),
#     work_queue_name="ml",
#     infrastructure=Process()
# )


if __name__ == "__main__":
    main_flow()
    #deployment.apply()
