# MLOps Lending Club Loan Prediction Project

This is the final project for the [MLOps ZoomCamp](https://github.com/DataTalksClub/mlops-zoomcamp) course  by [DataTalks.Club](https://datatalks.club/).  

## Problem Statement

We need to build an end-to-end machine learning system that predicts whether a particular user returns the loan or not. 
We access the probability of full repayment based on various features - employment length, annual income, home ownership status, etc. 

It should be a fault-tolerant, monitored, and containerized web service that receives the object's features and returns whether the loan will be repaid.

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
6. Open [Grafana](http://127.0.0.1:3000/) in the browser and find `Evidently Data Drift Dashboard`
7. Enjoy the live data drift detection!

### Experimentation and orchestration part

1. Set up the environment and prepare the project
```
make setup
```
2. Start Prefect server
```
pipenv run prefect orion start --host 0.0.0.0
```
3. Install [aws-cli](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and configure AWS profile

  * If you've already created an AWS account, head to the IAM section, generate your secret-key, and download it locally. 
  [Instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-prereqs.html)

  * [Configure](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html) `aws-cli` with your downloaded AWS secret keys:
      ```shell
         $ aws configure
         AWS Access Key ID [None]: xxx
         AWS Secret Access Key [None]: xxx
         Default region name [None]: eu-west-1
         Default output format [None]:
      ```

  * Verify aws config:
      ```shell
        $ aws sts get-caller-identity
      ```
4. Set S3 bucket
```
export BUCKET_NAME=s3-bucket-name
```
5. Run MLFlow server
```
pipenv run mlflow server --default-artifact-root s3://$BUCKET_NAME --backend-store-uri sqlite:///mlflow_db.sqlite
```
6. Create the deployment for the training pipeline
```
pipenv run python -m prediction_service.train_workflow
```
7. Run the deployment
```
pipenv run prefect deployment run 'main-flow/model_training_workflow'
```
8. Run the agent
```
pipenv run prefect agent start -q 'mlops'
```
9. Wait until it finishes and registers the new production model.


### Run tests and code quality checks

Unit tests
```
make test
```

Integration tests
```
make integration_test
```

Code formatting and code quality checks (`isort`, `black`, `pylint`)
```
make quality_checks
```

### Pre-commit hooks
Code formatting [pre-commit](.pre-commit-config.yaml) hooks are triggered on each commit 

### CI/CD
PR triggers [CI Workflow](.github/workflows/cd-tests.yaml)
* Environment setup, Unit tests, and Integration test
