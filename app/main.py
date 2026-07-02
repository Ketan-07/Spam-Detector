"""
FastAPI backend for the Spam Message Detector.

Serves:
  - POST /api/predict   -> classify a message as spam/ham
  - GET  /api/health     -> simple health check
  - /                     -> the static frontend (static/index.html)
"""

import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Spam Message Detector API")

# Allow the frontend to call the API even if it's ever hosted separately.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model file not found at {MODEL_PATH}. "
                "Run `python app/train_model.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)


class PredictResponse(BaseModel):
    label: str
    is_spam: bool
    confidence: float


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    model = get_model()
    label = model.predict([message])[0]

    confidence = 0.5
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([message])[0]
        classes = list(model.classes_)
        confidence = float(proba[classes.index(label)])

    return PredictResponse(
        label=label,
        is_spam=(label == "spam"),
        confidence=round(confidence, 4),
    )


# Serve the frontend last, so /api/* routes above take priority.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
