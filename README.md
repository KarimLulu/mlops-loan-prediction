# MLOps Lending Club Loan Prediction Project

This is the final project for the [MLOps ZoomCamp](https://github.com/DataTalksClub/mlops-zoomcamp) course  by [DataTalks.Club](https://datatalks.club/).  

## Problem Statement

We need to build the end-to-end machine learning system that predicts whether a particular user returns the loan or not. 
We access the probability of full repayment based on various features - employment length, annual income, home ownership status, etc. 

It should be a fault-tolerant, monitored, and containerized web service that receives object's features and returns whether the loan will be repaid.

## System Description
The system contains the following parts:
* [Experiment tracking](./prediction_service/train_workflow.py) and model [registry](./prediction_service/model_service.py) (we use `MLFlow`)
* An orchestrated model training [pipeline](./prediction_service/train_workflow.py) (`Prefect`)
* [Web service](./prediction_service/service.py) deployment (`Flask`)
* Model [monitoring](./evidently_service)
* Test coverage with [unit](./tests) and [integration](./integration_test) tests
* [Code linters and formatters](pyproject.toml)
* [Pre-commit hooks](.pre-commit-config.yaml)
* [Makefile](./Makefile)
* CI pipeline ([GitHub workflow](https://github.com/KarimLulu/mlops-loan-prediction/actions))


The project runs locally and uses AWS S3 to store model artifacts using `MLFlow`. It is containerized and can be easily deployed to the cloud.


## Dataset
We took the Lending Club [dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and reduced it to 10k records to speed up training and prototyping.
It contains 23 features and the target column `is_bad` - whether the loan was repaid or not. For this project, we use 3 features - employment length, annual income, and home ownership status.

## How to Run

### Serving Part

1. Clone the repo
```
git clone https://github.com/KarimLulu/mlops-loan-prediction.git
```
2. Navigate to the project folder
```
cd mlops-loan-prediction
```

3.  Build all required services
```
docker-compose build
```
4. Create and start containers
```
docker-compose up
```
5. Send some data records to the system in a separate terminal window:
```
make setup
pipenv run python -m monitoring.send_data
```
6. Open [Grafana](http://127.0.0.1:3000/) in the browser and find `Evidently Data Drift Dashboard`.
7. Enjoy the live data drift detection!
