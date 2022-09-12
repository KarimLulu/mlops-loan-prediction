import requests
from deepdiff import DeepDiff

payload = {'home_ownership': 'MORTGAGE', 'emp_length': 2, 'annual_inc': 150000}

url = 'http://localhost:9696/predict'
actual_response = requests.post(url, json=payload).json()
expected_response = {
    'is_bad_loan': 0,
    'model_version': 'test-loan-predictor/production',
}

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff
