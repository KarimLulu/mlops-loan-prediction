import os

import mlflow
from flask import Flask, request, jsonify

from model_service import init_model
from settings import TRACKING_URI


mlflow.set_tracking_uri(TRACKING_URI)
MODEL_STAGE = os.getenv('MODEL_STAGE', 'production')
MLFLOW_MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'loan-predictor')
model_service = init_model(
    stage=MODEL_STAGE,
    mlflow_model_name=MLFLOW_MODEL_NAME
)
app = Flask('loan-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    payload = request.get_json()
    pred = model_service.predict(payload)

    result = {
        'is_bad_loan': pred,
        'model_version': f"{MLFLOW_MODEL_NAME}/{MODEL_STAGE}"
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
