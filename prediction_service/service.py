import os

import mlflow
import requests
from flask import Flask, jsonify, request
from pymongo import MongoClient
from model_service import init_model

EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "http://127.0.0.1:9897")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://host.docker.internal:5000"
)
MODEL_STAGE = os.getenv('MODEL_STAGE', 'production')
MLFLOW_MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'loan-predictor')
TEST_RUN = os.getenv('TEST_RUN', 'False') == 'True'

mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_service = init_model(stage=MODEL_STAGE, mlflow_model_name=MLFLOW_MODEL_NAME)
app = Flask('loan-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    payload = request.get_json()
    pred = model_service.predict(payload)

    result = {
        'is_bad_loan': pred,
        'model_version': f"{MLFLOW_MODEL_NAME}/{MODEL_STAGE}",
    }
    if not TEST_RUN:
        save_to_db(payload, pred)
        send_to_evidently_service(payload, pred)
    return jsonify(result)


def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/loan", json=[rec])


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
