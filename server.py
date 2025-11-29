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

# ------------------------------
# BASIC CONFIG
# ------------------------------
logging.basicConfig(level=logging.INFO)

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
        return pickle.load(f)

model = load_model()

# ------------------------------
# PREDICT ENDPOINT
# ------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Load image in the same mode used for training
        # (try "L" for grayscale; change to "RGB" if your model used color images)
        img = Image.open(io.BytesIO(contents)).convert("L")

        # Resize to expected size - change if training used a different size
        img = img.resize((32, 32))

        # Flatten
        arr = np.array(img).flatten().astype(np.float64)  # use float64 for sklearn

        # Determine expected feature length from model if possible
        expected_len = getattr(model, "n_features_in_", None)
        if expected_len is None:
            # fallback: if model has attribute `coef_` or similar, try to infer
            try:
                expected_len = model.coef_.shape[1]
            except Exception:
                expected_len = arr.size  # last resort

        if arr.size != expected_len:
            logging.info(f"Input feature size ({arr.size}) != model expects ({expected_len}). "
                         "Padding or trimming the input to match the model.")

        # Pad or trim to expected_len (pad with zeros)
        if arr.size < expected_len:
            padded = np.zeros(expected_len, dtype=np.float64)
            padded[:arr.size] = arr
            arr = padded
        elif arr.size > expected_len:
            arr = arr[:expected_len]

        # Normalize the same way as during training (0-1 here)
        arr = arr / 255.0

        X = arr.reshape(1, -1)

        # DEBUG: log some helpful diagnostics
        logging.info(f"X shape: {X.shape}, dtype: {X.dtype}, min: {X.min():.4f}, max: {X.max():.4f}")
        logging.info(f"Model type: {type(model)}, has_predict_proba: {hasattr(model, 'predict_proba')}")

        # Try predict_proba (top 3)
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
        logging.info(f"Raw model output (predict): {raw_pred!r}")

        # If model.predict returns integers (class indices), map to CLASS_LABELS
        if isinstance(raw_pred, (int, np.integer)):
            pred_index = int(raw_pred)
            if 0 <= pred_index < len(CLASS_LABELS):
                prediction = CLASS_LABELS[pred_index]
            else:
                prediction = "Unknown"
        else:
            # Model probably returns string labels already
            prediction = str(raw_pred)

        return {"prediction": prediction, "top": top}

    except Exception as e:
        logging.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ------------------------------
# RUN SERVER
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
