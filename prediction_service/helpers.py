import numpy as np
from sklearn.metrics import f1_score


def lgb_f1_score(y_true, y_pred):
    y_hat = np.where(y_pred < 0.5, 0, 1)
    return 'f1', f1_score(y_true, y_hat), True
