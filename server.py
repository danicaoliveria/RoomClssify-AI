# server.py
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

# ------------------------------
# BASIC CONFIG
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

app = FastAPI()

# ------------------------------
# CORS
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# STATIC FOLDERS
# ------------------------------
# adjust these directories if needed
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/Components", StaticFiles(directory="Components"), name="components")

# ------------------------------
# HTML PAGES
# ------------------------------
@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/train")
def train_page():
    return FileResponse("train.html")

# ------------------------------
# CLASS LABELS (EXACT ORDER)
# ------------------------------
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

# ------------------------------
# LOAD MODEL
# ------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found at: " + MODEL_PATH)

    with open(MODEL_PATH, "rb") as f:
        mdl = pickle.load(f)
    return mdl

model = load_model()

# Print model diagnostics on startup
def inspect_model(m):
    try:
        info = {
            "type": str(type(m)),
            "has_predict_proba": hasattr(m, "predict_proba"),
            "n_features_in_": getattr(m, "n_features_in_", None),
            "classes_": getattr(m, "classes_", None),
        }
        logging.info("Model loaded. Info: %s", info)
        return info
    except Exception:
        logging.exception("Model inspection failed")
        return None

inspect_model(model)

# ------------------------------
# Helper: get expected features count
# ------------------------------
def get_expected_features(m):
    # Prefer n_features_in_ if present
    n = getattr(m, "n_features_in_", None)
    if n is not None:
        return int(n)
    # If it's a sklearn Pipeline, try the final estimator
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(m, Pipeline):
            last = m.steps[-1][1]
            n = getattr(last, "n_features_in_", None)
            if n is not None:
                return int(n)
    except Exception:
        pass
    # Try to infer from coef_ if available (for linear models)
    try:
        coef = getattr(m, "coef_", None)
        if coef is not None:
            return int(coef.shape[1])
    except Exception:
        pass
    # Last resort: assume 1024 (32x32)
    return 32 * 32

# Cache expected len
EXPECTED_FEATURES = get_expected_features(model)
logging.info("Model expected input features: %s", EXPECTED_FEATURES)

# ------------------------------
# PREDICT ENDPOINT
# ------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Use grayscale if your model was trained that way.
        # If your training used RGB, change "L" -> "RGB" and reshape accordingly.
        img = Image.open(io.BytesIO(contents)).convert("L")
        img = img.resize((32, 32))

        arr = np.array(img).flatten().astype(np.float64)  # use float64 for sklearn
        original_len = arr.size

        # If the model expects fewer or more features, trim or pad
        if original_len != EXPECTED_FEATURES:
            logging.info("Input flattened length = %s, model expects = %s", original_len, EXPECTED_FEATURES)
            if original_len > EXPECTED_FEATURES:
                arr = arr[:EXPECTED_FEATURES]
                logging.info("Trimmed input from %s -> %s", original_len, EXPECTED_FEATURES)
            else:
                padded = np.zeros(EXPECTED_FEATURES, dtype=np.float64)
                padded[:original_len] = arr
                arr = padded
                logging.info("Padded input from %s -> %s", original_len, EXPECTED_FEATURES)

        # Normalize (0-1). If training used a different normalization, replicate that here.
        arr = arr / 255.0
        X = arr.reshape(1, -1)

        logging.info("X shape: %s dtype: %s min: %.4f max: %.4f", X.shape, X.dtype, X.min(), X.max())

        # Try predict_proba for top-3
        top = []
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X)[0]
                idx = np.argsort(probs)[::-1][:3]
                top = [{"label": (CLASS_LABELS[i] if i < len(CLASS_LABELS) else str(i)),
                        "prob": float(probs[i])} for i in idx]
            except Exception:
                logging.exception("predict_proba failed")

        # Run predict
        raw_pred = model.predict(X)[0]
        logging.info("Raw model output: %r", raw_pred)

        # Map numeric indices to class labels if necessary
        if isinstance(raw_pred, (int, np.integer)):
            idx = int(raw_pred)
            if 0 <= idx < len(CLASS_LABELS):
                prediction = CLASS_LABELS[idx]
            else:
                prediction = "Unknown"
        else:
            # model may return string labels already
            prediction = str(raw_pred)

        return {"prediction": prediction, "top": top}

    except Exception as e:
        logging.exception("Prediction failed")
        # Include the message but keep stack trace only in logs
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# ------------------------------
# RUN SERVER
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
