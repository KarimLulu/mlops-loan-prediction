import logging
from pathlib import Path


def setup_logging(**kwargs):
    logging.basicConfig(**kwargs)


def init_dirs(*dirs):
    for folder in dirs:
        Path(folder).mkdir(parents=True, exist_ok=True)


# Common paths
REPO_DIR = Path(__file__).parent
DATA_DIR = REPO_DIR / "data"
DATA_NAME = "Lending_Club_reduced.csv"
DATA_PATH = DATA_DIR / DATA_NAME
init_dirs(DATA_DIR)

# Logging settings
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s - %(message)s"
LOG_DATE_FORMAT = "%d-%m-%Y %H:%M:%S"
LOG_LEVEL = logging.INFO


# Model settings
EXCLUDE_COLUMNS = ["Id"]
LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "min_data_in_leaf": 20,
    "min_gain_to_split": 0.0,
    "subsample_freq": 5,
    "subsample": 0.8,
    "feature_fraction": 0.9,
    "feature_fraction_bynode": 0.9,
    "max_bin": 255,
    "learning_rate": 0.1,
    "n_estimators": 50,
    "reg_alpha": 1e-1,
    "reg_lambda": 1e-1,
    "random_state": 30,
    "seed": 30,
    "metric": "",
    "n_jobs": -1,  # does not influence reproducibility in the latest version.
    "verbose": -1,
    "first_metric_only": False
}

GRID_SEARCH_PARAMS = {
    "lgbmclassifier__reg_alpha": [1e-2],
    "lgbmclassifier__reg_lambda": [1e-2],
}
EXPERIMENT_NAME = "loan-prediction"
TRACKING_URI = "http://host.docker.internal:5000"


try:
    from local_settings import *
except ImportError:
    pass

setup_logging(level=LOG_LEVEL,
              format=LOG_FORMAT,
              datefmt=LOG_DATE_FORMAT)