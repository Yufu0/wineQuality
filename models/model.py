from sklearn.ensemble import RandomForestClassifier

from models.dataset import Wine


class ModelRandomForest:
    def __init__(self):
        self.model = RandomForestClassifier(
            max_depth=7,
            n_estimators=10,
            max_features=7,
            random_state=42
        )
        self.parameters = {
            "max_depth": 7,
            "n_estimators": 10,
            "max_features": 7,
            "random_state": 42
        }

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, dataset):
        X, y = dataset.get_data()
        return self.model.score(X, y)

    def get_params(self):
        return self.parameters


class ModelWrapper:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def add_data(self, wine: Wine):
        self.dataset.add_data(wine)

    def fit_model(self):
        X, y = self.dataset.get_data()
        self.model.fit(X, y)

    def predict(self, wine: Wine):
        x = self.dataset.wine_to_x(wine)
        print(x)
        return self.model.predict(x)

    def score(self):
        return self.model.score(self.dataset)

    def get_params(self):
        return self.model.get_params()

    def get_info(self):
        return "Modèle de prédiction de la qualité du vin"


