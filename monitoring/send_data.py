import json
import pandas as pd
from time import sleep

from prediction_service.settings import DATA_PATH

import requests

data = pd.read_csv(DATA_PATH)

with open("target.csv", 'w') as f_target:
    for _, row in data.iterrows():
        payload = {
            'home_ownership': row['home_ownership'],
            'emp_length': row['emp_length'],
            'annual_inc': row['annual_inc']
        }
        resp = requests.post("http://127.0.0.1:9696/predict",
                             headers={"Content-Type": "application/json"},
                             data=json.dumps(payload)).json()
        print(f"Prediction: {resp['is_bad_loan']}")
        sleep(0.05)
