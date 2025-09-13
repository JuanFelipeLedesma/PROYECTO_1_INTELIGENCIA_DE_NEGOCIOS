# src/make_cv_reports.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from pipelines import build_pipeline

TRAIN = Path("data/Datos_proyecto.xlsx")
SHEET = 0
COL_TXT = "textos"
COL_Y   = "labels"
MAP = {1:"ODS1", 3:"ODS3", 4:"ODS4"}  # mapeo usado

df = pd.read_excel(TRAIN, sheet_name=SHEET, engine="openpyxl")
df[COL_TXT] = df[COL_TXT].fillna("").astype(str).str.strip()
y = df[COL_Y].map(MAP).astype(str)
X = df[COL_TXT].tolist()

pipe = build_pipeline(model="svm")  # ganador
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Predicciones out-of-fold (más honestas que en train)
y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")

labels = sorted(y.unique().tolist())
rep_txt = classification_report(y, y_pred, labels=labels, digits=4)
cm = confusion_matrix(y, y_pred, labels=labels)

outdir = Path("evaluation"); outdir.mkdir(exist_ok=True)
(outdir/"cv_classification_report.txt").write_text(rep_txt, encoding="utf-8")
pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                columns=[f"pred_{l}" for l in labels]).to_csv(outdir/"cv_confusion_matrix.csv", encoding="utf-8")
print("== CV REPORT ==\n", rep_txt)
print("Guardado: evaluation/cv_classification_report.txt y cv_confusion_matrix.csv")
