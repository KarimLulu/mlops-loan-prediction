import requests

payload = {
        'home_ownership': 'MORTGAGE',
        'emp_length': 2,
        'annual_inc': 150000
    }

url = 'http://localhost:9696/predict'
response = requests.post(url, json=payload)
print(response.json())
