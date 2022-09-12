import os

import mlflow


def get_model_location(stage='Production', mlflow_model_name='loan-predictor'):
    model_location = os.getenv('MODEL_LOCATION')

    if model_location is not None:
        return model_location

    model_uri = f"models:/{mlflow_model_name}/{stage}"
    return model_uri


def load_model(stage='Production', mlflow_model_name='loan-predictor'):
    model_path = get_model_location(stage=stage, mlflow_model_name=mlflow_model_name)
    model = mlflow.pyfunc.load_model(model_uri=model_path)
    return model


class ModelService:
    def __init__(self, model, model_version=None):
        self.model = model
        self.model_version = model_version

    @staticmethod
    def prepare_features(payload):
        features = {
            'home_ownership': payload['home_ownership'],
            'emp_length': payload['emp_length'],
            'annual_inc': payload['annual_inc'],
        }
        return features

    def predict(self, payload):
        features = self.prepare_features(payload)
        preds = self.model.predict(features)
        return int(preds[-1])


def init_model(stage: str, mlflow_model_name: str):
    model = load_model(stage, mlflow_model_name)
    model_service = ModelService(model=model, model_version=stage)
    return model_service
