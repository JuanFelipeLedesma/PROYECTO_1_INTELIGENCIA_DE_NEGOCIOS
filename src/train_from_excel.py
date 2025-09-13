# src/train_from_excel.py
import os, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pipelines import build_pipeline, evaluate_model
import joblib

TRAIN_PATH = Path("data/Datos_proyecto.xlsx")
SHEET_NAME = 0                 # 'Sheet1' en tu caso
COL_TXT    = "textos"          # <- tu columna de texto en TRAIN
COL_Y      = "labels"          # <- tu columna de etiqueta en TRAIN (numérica)

# 1) Cargar datos
df = pd.read_excel(TRAIN_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
df[COL_TXT] = df[COL_TXT].fillna("").astype(str).str.strip()
y_raw = df[COL_Y]

# 2) Mapear etiquetas numéricas a ODS si aplica (1->ODS1, 3->ODS3, 4->ODS4)
unique = set(pd.unique(y_raw))
mapping = {1:"ODS1", 3:"ODS3", 4:"ODS4"}
if unique.issubset(set(mapping.keys())):
    y = y_raw.map(mapping).astype(str)
    print(f"[INFO] Mapeo aplicado a ODS: {mapping}")
else:
    y = y_raw.astype(str)
    print("[WARN] Etiquetas no eran 1/3/4 puras; se usarán como string sin mapear.")

X = df[COL_TXT].tolist()

# 3) Probar 3 modelos y elegir el mejor por F1 macro (CV 5-fold estratificado)
candidates = ["svm", "nb", "lr"]  # LinearSVM, NaiveBayes, LogisticRegression
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = {}
for model_name in candidates:
    pipe = build_pipeline(model=model_name)
    f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
    scores[model_name] = {"mean_f1_macro": float(np.mean(f1)), "folds": f1.tolist()}
    print(f"[CV] {model_name}: F1_macro = {np.mean(f1):.4f}  folds={np.round(f1,4)}")

best = max(scores.items(), key=lambda kv: kv[1]["mean_f1_macro"])[0]
print(f"\n[WINNER] Mejor modelo por CV: {best}  (F1_macro={scores[best]['mean_f1_macro']:.4f})")

# 4) Entrenar el mejor en TODO el train y guardar
best_pipe = build_pipeline(model=best)
best_pipe.fit(X, y)

os.makedirs("src/models", exist_ok=True)
joblib.dump(best_pipe, "src/models/model.joblib")
print("[SAVE] Modelo guardado en src/models/model.joblib")

# 5) Métrica de referencia en train completo (solo informativa; confía más en el CV)
train_metrics = evaluate_model(best_pipe, X, y)
os.makedirs("evaluation", exist_ok=True)
with open("evaluation/model_selection.json", "w", encoding="utf-8") as f:
    json.dump({"scores_cv": scores, "winner": best, "train_metrics": train_metrics}, f, indent=2, ensure_ascii=False)
print("[EVAL] Métricas train:", train_metrics)
print("[OK] Listo.")
