# Proyecto 1 – Analítica de Textos (Starter Kit)

Este kit te da una base **lista para empezar**: plantillas, contratos de API, esqueletos de pipeline y wiki.

## Cómo usarlo (resumen)
1. Crea tu entorno:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Entrena un primer modelo de baseline:
   ```bash
   python src/train.py
   ```
   - Guarda el modelo en `src/models/model.joblib` (el script ya está preparado).
3. Sirve la API (FastAPI):
   ```bash
   uvicorn api.main:app --reload
   ```
   - `/docs` expone Swagger para probar **/predict** y **/retrain**.
4. Llena el **Canvas**, **Report** y **Wiki** con las plantillas en `docs/` y `wiki/`.
5. Usa `excel/plantilla_test.csv` para la **entrega de Etapa 1** (agrega tu columna de predicción).

## Estructura
```
api/              # FastAPI con endpoints /predict y /retrain
schema/           # JSON Schemas para requests
src/              # Pipelines: datos, entrenamiento, predicción, reentrenamiento
docs/             # Plantillas (Canvas, Reporte, Video script, Checklists)
wiki/             # Skeleton para Wiki del repo
excel/            # Plantilla CSV de test
evaluation/       # Template de métricas
data/             # Ejemplos de entrada
requirements.txt  # Dependencias mínimas
```

## Recomendación
- Empezar con **TF-IDF + LinearSVM / Naive Bayes / Logistic Regression**.
- Reportar **precision/recall/F1 por clase** y **macro-F1**.
- Explicar **palabras/características** que disparan cada ODS.
