# inspect_columns.py
import pandas as pd
from pathlib import Path

TRAIN = Path("data/Datos_proyecto.xlsx")                 # ← tu train
TEST  = Path("data/Datos de prueba_proyecto.xlsx")       # ← tu test

def show_book(path: Path):
    print("\n" + "="*80)
    print(f"Archivo: {path}")
    xls = pd.ExcelFile(path)
    print("Hojas encontradas:", xls.sheet_names)
    for sheet in xls.sheet_names:
        print("-"*80)
        print(f"[Hoja] {sheet}")
        df = pd.read_excel(path, sheet_name=sheet)
        print("Shape:", df.shape)
        print("Columnas:", list(df.columns))
        # Muestra 5 filas para que identifiques nombres reales de columnas
        print(df.head(5))
        # Info útil para detectar la columna de texto y la etiqueta
        print("\nTipos:", df.dtypes.to_dict())
        print("Nulos por columna:", df.isna().sum().to_dict())
        # Si es muy grande, corta
        break  # ← quita este 'break' si quieres imprimir TODAS las hojas completas

def main():
    show_book(TRAIN)
    show_book(TEST)

if __name__ == "__main__":
    main()
