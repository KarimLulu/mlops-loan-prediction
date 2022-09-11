import os

import mlflow
from flask import Flask, request, jsonify


RUN_ID = os.getenv('RUN_ID', 'df5973fcf13a45e7a7a04136603bc8cc')
BUCKET_NAME = os.getenv('BUCKET_NAME', 'mlops-loan-prediction')

logged_model = f's3://{BUCKET_NAME}/1/{RUN_ID}/artifacts/model'
print(logged_model)
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(payload):
    features = {
        'home_ownership': payload['home_ownership'],
        'emp_length': payload['emp_length'],
        'annual_inc': payload['annual_inc']
    }
    return features


def predict(features):
    preds = model.predict(features)
    return int(preds[-1])


app = Flask('loan-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    payload = request.get_json()

    features = prepare_features(payload)
    pred = predict(features)

    result = {
        'is_bad_loan': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
