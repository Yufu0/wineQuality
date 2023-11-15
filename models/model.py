from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class ModelRandomForest:
    def __init__(self):
        self.model = RandomForestClassifier(
            max_depth=7,
            n_estimators=10,
            max_features=7,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        self.model.predict(X)


class ModelWrapper:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def add_data(self, x, y):
        self.dataset.add_data(x, y)

    def fit_model(self):
        X, y = self.dataset.get_data()
        self.model.fit(X, y)

    def predict(self, X):
        self.model.predict(X)


