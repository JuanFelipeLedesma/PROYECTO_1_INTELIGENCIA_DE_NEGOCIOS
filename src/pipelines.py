from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
import numpy as np

# Elige tu modelo base aquí (rápido y efectivo para baseline)
BASE_MODEL = "svm"  # opciones: "svm", "nb", "lr"

def build_pipeline(model: str = BASE_MODEL) -> Pipeline:
    if model == "svm":
        clf = LinearSVC()
    elif model == "nb":
        clf = MultinomialNB()
    elif model == "lr":
        clf = LogisticRegression(max_iter=200)
    else:
        raise ValueError("Modelo no soportado")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2), strip_accents="unicode", stop_words="spanish")),
        ("clf", clf)
    ])
    return pipe

def evaluate_model(model: Pipeline, X, y):
    y_pred = model.predict(X)
    labels = sorted(list(set(y)))
    p, r, f1, support = precision_recall_fscore_support(y, y_pred, labels=labels, zero_division=0)
    macro_f1 = f1_score(y, y_pred, average="macro")
    report = {
        "precision_macro": float(np.mean(p)),
        "recall_macro": float(np.mean(r)),
        "f1_macro": float(macro_f1),
        "per_class": {lbl: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])} for i, lbl in enumerate(labels)}
    }
    return report
