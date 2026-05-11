import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE, "model_rf.pkl")
DATA_PATH = os.path.join(_BASE, "data", "dataset.csv")

_cached_model = None


def train_model():
    data = pd.read_csv(DATA_PATH)
    X = data[["age", "income", "loan", "credit_score", "emi"]]
    y = data["risk"]

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    model.fit(X, y)

    scores = cross_val_score(model, X, y, cv=min(5, len(data) // 2))
    print(f"[model] CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    joblib.dump(model, MODEL_PATH)
    return model


def predict(features: list) -> int:
    global _cached_model
    if _cached_model is None:
        try:
            _cached_model = joblib.load(MODEL_PATH)
        except (FileNotFoundError, Exception):
            _cached_model = train_model()
    try:
        return int(_cached_model.predict([features])[0])
    except Exception:
        return 0


def get_feature_importances() -> dict:
    global _cached_model
    if _cached_model is None:
        try:
            _cached_model = joblib.load(MODEL_PATH)
        except Exception:
            _cached_model = train_model()
    names = ["Age", "Income", "Loan", "Credit Score", "EMI"]
    return dict(zip(names, _cached_model.feature_importances_))
