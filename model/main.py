import logging
import pandas as pd
import sys

from model.estimator import Estimator
from settings import DATA_PATH


logger = logging.getLogger(__name__)


def main():
    model = Estimator()
    data = pd.read_csv(DATA_PATH)
    X = data.loc[:, data.columns != "is_bad"]
    y = data["is_bad"].values
    params = model.tune_parameters(X, y, random_state=42,
                                   encoder_params={"random_state": 42})
    logger.info(params["scores"])
    return 0


if __name__ == "__main__":
    code = main()
    sys.exit(code)
