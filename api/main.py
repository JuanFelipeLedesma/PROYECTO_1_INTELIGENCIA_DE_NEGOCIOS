from fastapi import FastAPI, HTTPException
from .schemas import PredictRequest, PredictResponse, RetrainRequest, RetrainResponse
from typing import List, Dict
import joblib
import os
import time

app = FastAPI(title="UNFPA ODS Classifier API", version="0.1.0")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "models", "model.joblib")

def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}. Entrena primero con src/train.py")
    return joblib.load(MODEL_PATH)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = _load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Extraer textos
    textos = [inst.texto for inst in req.instances]
    # predict / predict_proba
    try:
        y_pred = model.predict(textos).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")

    resp = PredictResponse(predictions=y_pred)
    # Si el modelo soporta predict_proba, devolver probabilidades
    if hasattr(model, "predict_proba"):
        try:
            import numpy as np
            probs = model.predict_proba(textos)
            classes = list(model.classes_)
            prob_dicts: List[Dict[str, float]] = []
            for row in probs:
                prob_dicts.append({cls: float(p) for cls, p in zip(classes, row)})
            resp.probabilities = prob_dicts
        except Exception:
            pass

    return resp

@app.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest):
    # Ejemplo simple: reentrenar desde cero con TF-IDF + modelo base
    try:
        from ..src.pipelines import build_pipeline, evaluate_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo importar pipeline: {e}")

    textos = [inst.texto for inst in req.instances]
    labels = req.labels
    if len(textos) != len(labels):
        raise HTTPException(status_code=400, detail="instances y labels deben tener la misma longitud")

    try:
        pipe = build_pipeline()
        pipe.fit(textos, labels)
        # Persistir
        os.makedirs(os.path.join(os.path.dirname(__file__), "..", "src", "models"), exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
        # Evaluación mínima (hold-out pequeño puede ser agregado por ustedes)
        metrics = evaluate_model(pipe, textos, labels)
        version = time.strftime("%Y%m%d-%H%M%S")
        return RetrainResponse(metrics=metrics, model_version=version)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en reentrenamiento: {e}")
