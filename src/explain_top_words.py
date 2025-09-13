from pathlib import Path
import numpy as np, joblib

TOPN = 25
MODEL = Path("src/models/model.joblib")
pipe = joblib.load(MODEL)
vec = pipe.named_steps["tfidf"]
clf = pipe.named_steps["clf"]

feature_names = vec.get_feature_names_out()
classes = clf.classes_
coefs = clf.coef_  # LinearSVC (one-vs-rest)

lines = []
for i, cls in enumerate(classes):
    idx = np.argsort(coefs[i])[-TOPN:][::-1]
    terms = feature_names[idx]
    weights = coefs[i][idx]
    lines.append(f"\n=== TOP {TOPN} términos para {cls} ===")
    lines += [f"{t}\t{w:.4f}" for t, w in zip(terms, weights)]

out = Path("evaluation/top_words.txt")
out.write_text("\n".join(lines), encoding="utf-8")
print("Guardado:", out)
