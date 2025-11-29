# Rewritten server.py (Clean, Predicts: Bedroom, Office, etc.)

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
# CONFIG
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

app = FastAPI()

# ----------------------------------------------------
# CORS
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# STATIC FILES
# ----------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/Components", StaticFiles(directory="Components"), name="components")

# ----------------------------------------------------
# HTML ROUTES
# ----------------------------------------------------
@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/train")
def train_page():
    return FileResponse("train.html")

# ----------------------------------------------------
# CLASS LABELS (FINAL OUTPUTS)
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
# LOAD MODEL
# ----------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ----------------------------------------------------
# INSPECT MODEL
# ----------------------------------------------------
def inspect_model(m):
    try:
        info = {
            "type": str(type(m)),
            "has_predict_proba": hasattr(m, "predict_proba"),
            "n_features_in_": getattr(m, "n_features_in_", None),
            "classes_": getattr(m, "classes_", None),
        }
        logging.info(f"Model loaded: {info}")
        return info
    except Exception:
        logging.exception("Model inspection failed")
        return None

inspect_model(model)

# ----------------------------------------------------
# EXPECTED FEATURES
# ----------------------------------------------------
def get_expected_features(m):
    n = getattr(m, "n_features_in_", None)
    if n is not None:
        return int(n)

    try:
        from sklearn.pipeline import Pipeline
        if isinstance(m, Pipeline):
            last = m.steps[-1][1]
            n = getattr(last, "n_features_in_", None)
            if n:
                return int(n)
    except:
        pass

    coef = getattr(m, "coef_", None)
    if coef is not None:
        return int(coef.shape[1])

    return 32 * 32

EXPECTED_FEATURES = get_expected_features(model)
logging.info(f"Model expects input size: {EXPECTED_FEATURES}")

# ----------------------------------------------------
# PREDICT
# ----------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L")  # grayscale
        img = img.resize((32, 32))

        arr = np.array(img).flatten().astype(np.float64)
        original_len = arr.size

        # Match model input size
        if original_len != EXPECTED_FEATURES:
            if original_len > EXPECTED_FEATURES:
                arr = arr[:EXPECTED_FEATURES]
            else:
                padded = np.zeros(EXPECTED_FEATURES, dtype=np.float64)
                padded[:original_len] = arr
                arr = padded

        # Normalize
        arr = arr / 255.0
        X = arr.reshape(1, -1)

        # Try top-3
        top = []
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X)[0]
                idx = np.argsort(probs)[::-1][:3]
                top = [{
                    "label": CLASS_LABELS[i] if i < len(CLASS_LABELS) else str(i),
                    "prob": float(probs[i])
                } for i in idx]
            except:
                logging.exception("predict_proba failed")

        # Final prediction
        raw = model.predict(X)[0]

        if isinstance(raw, (int, np.integer)) and 0 <= raw < len(CLASS_LABELS):
            prediction = CLASS_LABELS[int(raw)]
        else:
            prediction = str(raw)

        return {
            "prediction": prediction,
            "top": top
        }

    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# ----------------------------------------------------
# RUN SERVER
# ----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)