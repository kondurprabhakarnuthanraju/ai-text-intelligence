import numpy as np

class Ensemble:
    def __init__(self, fitted_models):
        self.models = fitted_models
        self.classes_ = np.array([0, 1])

    def predict_proba(self, texts):
        probabilities = [
            model.predict_proba(texts)
            for model in self.models.values()
        ]
        return np.mean(probabilities, axis=0)

    def predict(self, texts):
        return (
            self.predict_proba(texts)[:, 1] >= 0.5
        ).astype(int)
