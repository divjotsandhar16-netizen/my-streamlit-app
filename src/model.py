# src/model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import numpy as np

def build_default_model(**kwargs):
    return RandomForestClassifier(n_estimators=150, random_state=42, **kwargs)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm, "preds": preds}

def save_model(model, scaler, columns, path_model="model.joblib"):
    """
    Save model + scaler + column order
    """
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "columns": list(columns)
    }, path_model)

def load_model(path_model="model.joblib"):
    data = joblib.load(path_model)
    return data["model"], data["scaler"], data["columns"]

def predict(model, scaler, X_raw):
    """
    Predict using trained model.
    If scaler exists, apply scaling.
    """
    if scaler is not None:
        X_scaled = scaler.transform(X_raw)
    else:
        X_scaled = X_raw

    preds = model.predict(X_scaled)
    return preds
