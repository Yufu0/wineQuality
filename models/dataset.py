from typing import Optional

import pandas as pd
import numpy as np
from pydantic import BaseModel


class Wine(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    quality: Optional[int] = None


class DatasetWineQuality:
    def __init__(self):
        self.columns = []
        self.X = np.array([])
        self.y = np.array([])

    def add_data(self, wine: Wine):
        self.X = np.append(
            self.X,
            np.array([[
                wine.fixed_acidity,
                wine.volatile_acidity,
                wine.citric_acid,
                wine.residual_sugar,
                wine.chlorides,
                wine.free_sulfur_dioxide,
                wine.total_sulfur_dioxide,
                wine.density,
                wine.pH,
                wine.sulphates,
                wine.alcohol
            ]]),
            axis=0
        )
        self.y = np.append(self.y, [np.array(wine.quality)], axis=0)

    def get_data(self):
        return self.X, self.y

    def load_data(self, path: str):
        data = pd.read_csv(path)
        self.columns = data.columns
        self.X = data.drop(columns=['quality', 'Id']).to_numpy()
        self.y = data['quality'].to_numpy()

    def wine_to_x(self, wine: Wine):
        return np.array([[
            wine.fixed_acidity,
            wine.volatile_acidity,
            wine.citric_acid,
            wine.residual_sugar,
            wine.chlorides,
            wine.free_sulfur_dioxide,
            wine.total_sulfur_dioxide,
            wine.density,
            wine.pH,
            wine.sulphates,
            wine.alcohol
        ]])
