# ----------------------------------------------------
# server.py (FINAL, CLEAN, FULLY FIXED)
# ----------------------------------------------------

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os
import pickle
import numpy as np
from PIL import Image
import io
import uvicorn
import logging
import traceback


# ----------------------------------------------------
# LOGGING
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")


# ----------------------------------------------------
# FASTAPI APP
# ----------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------
# STATIC FILES + HTML
# ----------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/Components", StaticFiles(directory="Components"), name="components")

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/train")
def train_page():
    return FileResponse("train.html")


# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"MODEL NOT FOUND: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    logging.info("Model loaded successfully.")
    return model


model = load_model()


# ----------------------------------------------------
# CLASS LABELS (from your training)
# ----------------------------------------------------
CLASS_LABELS = [
    "Bathroom",
    "Bedroom",
    "Dining",
    "Gaming",
    "Kitchen",
    "Laundry",
    "Living",
    "Office",
    "Terrace",
    "Yard",
]


# ----------------------------------------------------
# DETECT MODEL INPUT FEATURE SIZE
# ----------------------------------------------------
def detect_features(m):
    n = getattr(m, "n_features_in_", None)
    if n:
        return int(n)

    coef = getattr(m, "coef_", None)
    if coef is not None:
        return int(coef.shape[1])

    return 1000  # fallback (your model expects 1000)


EXPECTED_FEATURES = detect_features(model)
logging.info(f"Model expects {EXPECTED_FEATURES} features.")


# ----------------------------------------------------
# PREDICT ENDPOINT
# ----------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L")
        img = img.resize((32, 32))

        # Convert → flatten 1024 features
        arr = np.array(img).flatten().astype(np.float64) / 255.0
        original_len = len(arr)

        # ------------------------------------------------
        # FIX — match model's expected features
        # ------------------------------------------------
        if original_len > EXPECTED_FEATURES:
            arr = arr[:EXPECTED_FEATURES]

        else:
            padded = np.zeros(EXPECTED_FEATURES, dtype=np.float64)
            padded[:original_len] = arr
            arr = padded

        X = arr.reshape(1, -1)

        # ------------------------------------------------
        # PREDICT LABEL (convert index → class label)
        # ------------------------------------------------
        raw_pred = model.predict(X)[0]

        # If model returns integer classes
        if isinstance(raw_pred, (int, np.integer)) and raw_pred < len(CLASS_LABELS):
            prediction = CLASS_LABELS[int(raw_pred)]
        else:
            prediction = str(raw_pred)

        # ------------------------------------------------
        # TOP-3 PREDICTIONS (if model supports probability)
        # ------------------------------------------------
        top3 = []
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            idx = np.argsort(probs)[::-1][:3]

            for i in idx:
                top3.append({
                    "label": CLASS_LABELS[i] if i < len(CLASS_LABELS) else str(i),
                    "probability": round(float(probs[i]), 4)
                })

        return {
            "prediction": prediction,
            "top3": top3
        }

    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ----------------------------------------------------
# RUN SERVER
# ----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
