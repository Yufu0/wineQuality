from fastapi import FastAPI, HTTPException
import uvicorn
import os

from models.dataset import DatasetWineQuality, Wine
from models.model import ModelWrapper, ModelRandomForest

app = FastAPI()
model = None


@app.get("/api/model")
def get_model():
    if model is None:
        raise HTTPException(status_code=404, detail="Modèle non disponible")
    return {"model": "yep"}


@app.get("/api/model/description")
def get_model_description():
    if model is None:
        return {"message": "Model pas encore entrainé."}
    return {
        "parameters": model.get_params(),
        "metrics": {"accuracy": model.score()},
        "other_info": model.get_info()
    }


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


@app.get("/api/model/predict")
def predict_quality():
    if model is None:
        raise HTTPException(status_code=404, detail="Modèle non disponible")
    return {"best_wine": "yep"}


@app.post("/api/model/predict")
def predict_quality(wine: Wine):
    if wine.quality is not None:
        raise HTTPException(status_code=422, detail="Erreur format données")
    if model is None:
        raise HTTPException(status_code=404, detail="Modèle non disponible")

    prediction = model.predict(wine)
    return {"prediction": int(prediction[0])}


if __name__ == "__main__":

    model_forest = ModelRandomForest()
    dataset = DatasetWineQuality()
    dataset.load_data("data/Wines.csv")
    model = ModelWrapper(model_forest, dataset)
    model.fit_model()

    port = int(os.environ['FASTAPI_PORT'])

    uvicorn.run(app, host="0.0.0.0", port=port)
