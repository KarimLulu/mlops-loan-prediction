LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME:=loan-prediction-model:${LOCAL_TAG}

test:
	pipenv run pytest tests/

quality_checks:
	pipenv run isort .
	pipenv run black .
	pipenv run pylint --recursive=y prediction_service

build: quality_checks test
	docker build -t ${LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integration_test/run.sh

publish: integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh

setup:
	pipenv install --dev
	pipenv run pre-commit install
