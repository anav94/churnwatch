from fastapi import FastAPI
from typing import Dict, Any, List
import pandas as pd
import joblib, os

MODEL_PATH = os.getenv("CHURNWATCH_MODEL", "artifacts/model.joblib")
app = FastAPI(title="ChurnWatch Scoring API", version="1.0")
model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"ok": True, "message": "ChurnWatch API is running", "model_path": MODEL_PATH}

@app.post("/predict_one")
def predict_one(row: Dict[str, Any]):
    X = pd.DataFrame([row])
    proba = float(model.predict_proba(X)[:,1][0])
    return {"churn_probability": proba}

@app.post("/predict_batch")
def predict_batch(payload: Dict[str, Any]):
    rows: List[Dict[str, Any]] = payload.get("rows", [])
    X = pd.DataFrame(rows)
    probs = model.predict_proba(X)[:,1]
    return {"churn_probability": probs.tolist()}
