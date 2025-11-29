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
# SETTINGS
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
# STATIC FILES
# ------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/Components", StaticFiles(directory="Components"), name="components")

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/train")
def train_page():
    return FileResponse("train.html")

# ------------------------------
# CLASS LABELS (same order used during training)
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
        raise RuntimeError(f"Model not found at: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()
logging.info("Model loaded successfully.")

# Detect model input feature size
EXPECTED_FEATURES = getattr(model, "n_features_in_", None)
logging.info(f"Model expects {EXPECTED_FEATURES} features.")

# ------------------------------
# PROCESS IMAGE INTO FEATURES
# ------------------------------
def preprocess_image(contents):
    img = Image.open(io.BytesIO(contents))

    # ----- Detect whether model used RGB or Grayscale -----
    if EXPECTED_FEATURES == 3072:   # 32*32*3
        img = img.convert("RGB")
        img = img.resize((32, 32))
        arr = np.array(img).astype(np.float64).flatten()

    elif EXPECTED_FEATURES == 1024:  # 32*32*1
        img = img.convert("L")
        img = img.resize((32, 32))
        arr = np.array(img).astype(np.float64).flatten()

    else:
        raise ValueError(f"Unknown model feature size: {EXPECTED_FEATURES}")

    # Normalize
    arr = arr / 255.0

    # Final shape
    return arr.reshape(1, -1)


# ------------------------------
# PREDICT ENDPOINT
# ------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        X = preprocess_image(contents)

        # Raw prediction
        pred = model.predict(X)[0]
        logging.info(f"Raw prediction: {pred}")

        # If model already returns the string label
        if isinstance(pred, str):
            prediction = pred
        else:
            # If model returns index (integer)
            prediction = CLASS_LABELS[int(pred)]

        # ----- Top 3 -----
        top = []
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            idx = np.argsort(probs)[::-1][:3]
            top = [
                {"label": CLASS_LABELS[i], "prob": float(probs[i])}
                for i in idx
            ]

        return {
            "prediction": prediction,
            "top": top
        }

    except Exception as e:
        logging.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ------------------------------
# RUN SERVER
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
