from fastapi import FastAPI, HTTPException

from models.dataset import DatasetWineQuality, Wine
from models.model import ModelWrapper, ModelRandomForest

app = FastAPI()
model = None

model_description = {
    "parameters": {
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol"
    },
    "metrics": {
        "bof tier"
    },
    "other_info": {
        "Yasuo"
    }
}

#
# class Parameters(BaseModel):
#     fixed_acidity: float
#     volatile_acidity: float
#     citric_acid: float
#     residual_sugar: float
#     chlorides: float
#     free_sulfur_dioxide: float
#     total_sulfur_dioxide: float
#     density: float
#     pH: float
#     sulphates: float
#     alcohol: float

#
# class AdditionalData(BaseModel):
#     fixed_acidity: float
#     volatile_acidity: float
#     citric_acid: float
#     residual_sugar: float
#     chlorides: float
#     free_sulfur_dioxide: float
#     total_sulfur_dioxide: float
#     density: float
#     pH: float
#     sulphates: float
#     alcohol: float
#     quality: int
#
training_data = []


@app.get("/api/model")
def get_model():
    if model is None:
        raise HTTPException(status_code=404, detail="Modèle non disponible")
    return {"model": "yep"}


@app.get("/api/model/description")
def get_model_description():
    return {"parameters": model_description["parameters"],
            "metrics": model_description["metrics"],
            "other_info": model_description["other_info"]}


@app.put("/api/model")
def enrich_model(additional_data: Wine):
    if additional_data.quality is None:
        raise HTTPException(status_code=422, detail="Données d'entraînement incomplètes")
    if model is None:
        raise HTTPException(status_code=404, detail="Modèle non disponible")

    model.add_data(additional_data)

    return {"message": "Données d'entraînement ajoutées avec succès"}


@app.post("/api/model/retrain")
def retrain_model():
    if model is None:
        raise HTTPException(status_code=404, detail="Modèle non disponible")

    model.fit_model()

    return {"message": "Modèle réentraîné avec succès"}


@app.post("/api/model/predict")
def predict_quality(wine: Wine):
    if wine.quality is not None:
        raise HTTPException(status_code=422, detail="Erreur format données")
    if model is None:
        raise HTTPException(status_code=404, detail="Modèle non disponible")

    prediction = model.predict(wine)
    return {"prediction": int(prediction[0])}


if __name__ == "__main__":
    import uvicorn

    model_forest = ModelRandomForest()
    dataset = DatasetWineQuality()
    dataset.load_data("data/Wines.csv")
    model = ModelWrapper(model_forest, dataset)
    uvicorn.run(app, host="127.0.0.1", port=8000)

