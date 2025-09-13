# predict_excel.py
import sys
from pathlib import Path
import pandas as pd
import joblib

MODEL = Path("src/models/model.joblib")                # modelo ganador (SVM)
TEST  = Path("data/Datos de prueba_proyecto.xlsx")     # tu archivo de test
SHEET = 0                                              # 'Sheet1'
COL_TXT = "Textos_espanol"                             # columna de texto en TEST

def main():
    if not MODEL.exists():
        sys.exit(f"[ERROR] No encuentro el modelo en {MODEL}. Entrena primero.")

    print("[INFO] Cargando modelo…")
    m = joblib.load(MODEL)

    print(f"[INFO] Leyendo {TEST} (hoja={SHEET})…")
    df = pd.read_excel(TEST, sheet_name=SHEET, engine="openpyxl")

    if COL_TXT not in df.columns:
        sys.exit(f"[ERROR] No encuentro la columna '{COL_TXT}' en el test. "
                 f"Columnas disponibles: {list(df.columns)}")

    # normalización mínima y predicción
    df[COL_TXT] = df[COL_TXT].fillna("").astype(str).str.strip()
    df["prediccion_modelo"] = m.predict(df[COL_TXT])

    out = TEST.parent / "test_etiquetado.xlsx"
    df.to_excel(out, index=False)
    print(f"[OK] Archivo listo para entregar: {out}")
    print(df["prediccion_modelo"].value_counts())

if __name__ == "__main__":
    main()
