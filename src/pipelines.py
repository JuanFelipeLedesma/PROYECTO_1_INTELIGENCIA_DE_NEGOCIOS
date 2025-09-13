from typing import Iterable, Optional, Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np

# Elige tu modelo base aquí (rápido y efectivo para baseline)
BASE_MODEL = "svm"  # opciones: "svm", "nb", "lr"

def build_pipeline(
    model: str = BASE_MODEL,
    stopwords: Optional[Iterable[str]] = None,          # ← AHORA CONFIGURABLE (antes daba error con "spanish")
    ngram_range: Tuple[int, int] = (1, 2),
) -> Pipeline:
    """
    Construye un pipeline TF-IDF + clasificador.
    - model: "svm" (LinearSVC), "nb" (MultinomialNB), "lr" (LogisticRegression)
    - stopwords: None o lista de palabras a ignorar (para español, pásalas como lista)
    - ngram_range: por defecto (1,2)
    """
    if model == "svm":
        clf = LinearSVC()  # buen baseline para texto
    elif model == "nb":
        clf = MultinomialNB()
    elif model == "lr":
        clf = LogisticRegression(max_iter=1000)  # más iteraciones para converger con TF-IDF
    else:
        raise ValueError("Modelo no soportado: usa 'svm', 'nb' o 'lr'.")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram_range,
            strip_accents="unicode",
            stop_words=stopwords   # ← antes: "spanish" (NO válido en sklearn)
        )),
        ("clf", clf)
    ])
    return pipe

def evaluate_model(model: Pipeline, X: List[str], y: List[str]):
    """
    Devuelve métricas macro y por clase en dict:
    - precision_macro, recall_macro, f1_macro
    - per_class: {clase: {precision, recall, f1, support}}
    """
    y_pred = model.predict(X)
    labels = sorted(list(set(y)))
    p, r, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = f1_score(y, y_pred, average="macro")
    report = {
        "precision_macro": float(np.mean(p)),
        "recall_macro": float(np.mean(r)),
        "f1_macro": float(macro_f1),
        "per_class": {
            lbl: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            } for i, lbl in enumerate(labels)
        }
    }
    return report
