import os, joblib
from pipelines import build_pipeline, evaluate_model

# TODO: Reemplaza con tu carga real de datos
X_train = [
    "No hay acceso a salud en mi vereda",
    "El colegio no tiene suficientes profesores",
    "Mi familia no tiene ingresos suficientes para comer bien"
]
y_train = ["ODS3", "ODS4", "ODS1"]

if __name__ == "__main__":
    pipe = build_pipeline(model="svm")
    pipe.fit(X_train, y_train)
    os.makedirs("src/models", exist_ok=True)
    joblib.dump(pipe, "src/models/model.joblib")
    metrics = evaluate_model(pipe, X_train, y_train)
    print("Entrenado baseline. Métricas:", metrics)
