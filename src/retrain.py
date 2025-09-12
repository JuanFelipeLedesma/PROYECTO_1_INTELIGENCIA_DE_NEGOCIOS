import json, joblib, os
from pipelines import build_pipeline, evaluate_model

if __name__ == "__main__":
    # Espera un JSON con {"texts": [...], "labels": [...]}
    payload = json.loads(open("data/retrain_payload.json","r",encoding="utf-8").read())
    X, y = payload["texts"], payload["labels"]
    pipe = build_pipeline()
    pipe.fit(X, y)
    os.makedirs("src/models", exist_ok=True)
    joblib.dump(pipe, "src/models/model.joblib")
    print("Reentrenado OK. Métricas:", evaluate_model(pipe, X, y))
