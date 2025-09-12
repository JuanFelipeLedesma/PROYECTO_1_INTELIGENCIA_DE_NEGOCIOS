from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Instance(BaseModel):
    # Ajusta estos campos a tu dataset real:
    texto: str

class PredictRequest(BaseModel):
    instances: List[Instance]

class PredictResponse(BaseModel):
    predictions: List[str]
    # Opcional: probabilidades por clase
    probabilities: Optional[List[Dict[str, float]]] = None

class RetrainRequest(BaseModel):
    instances: List[Instance]
    labels: List[str]  # Etiquetas verdaderas para reentrenar

class RetrainResponse(BaseModel):
    metrics: Dict[str, Any]  # e.g., {"precision_macro": 0.81, "recall_macro": 0.79, "f1_macro": 0.80, "per_class": {...}}
    model_version: str
