import joblib, sys
model = joblib.load("src/models/model.joblib")
textos = sys.argv[1:] or ["Necesitamos más cupos en la escuela", "No tengo empleo estable"]
print(model.predict(textos).tolist())
