from prediction_service import model_service


def test_prepare_features():
    model = model_service.ModelService(model=None)

    payload = {
        'home_ownership': 'MORTGAGE',
        'emp_length': 10,
        'annual_inc': 150_000,
        'id': 10,
        'verif_status': 0,
    }

    actual_features = model.prepare_features(payload)

    expected_fetures = {
        'home_ownership': 'MORTGAGE',
        'emp_length': 10,
        'annual_inc': 150_000,
    }

    assert actual_features == expected_fetures


class ModelMock:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        n = len(X)
        return [self.value] * n


def test_predict():
    predicted_value = 1
    model_mock = ModelMock(predicted_value)
    model = model_service.ModelService(model_mock)

    payload = {
        'home_ownership': 'MORTGAGE',
        'emp_length': 10,
        'annual_inc': 150_000,
        'id': 10,
        'verif_status': 0,
    }

    actual_prediction = model.predict(payload)
    expected_prediction = predicted_value

    assert actual_prediction == expected_prediction
