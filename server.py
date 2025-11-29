# -------------------------------------------------------------
# RoomClassify FastAPI Server (CLEAN + FIXED VERSION)
# -------------------------------------------------------------
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import numpy as np
from PIL import Image
import pickle
import logging
import io
import os
import uvicorn

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static pages
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/Components", StaticFiles(directory="Components"), name="components")


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/train")
def train_page():
    return FileResponse("train.html")

# -------------------------------------------------------------
# ROOM LABELS (must match model training order)
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    logging.info(f"Model loaded: {type(model)}")
    return model


model = load_model()


# -------------------------------------------------------------
# MAIN PREDICT ENDPOINT
# -------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        # Convert to grayscale (MUST match training)
        img = Image.open(io.BytesIO(contents)).convert("L")
        img = img.resize((32, 32))

        # Flatten to vector
        arr = np.array(img).flatten().astype(np.float64)
        arr = arr / 255.0  # normalization from training

        # Make 2D for model
        X = arr.reshape(1, -1)

        # --------------------------
        # Predict
        # --------------------------
        raw_pred = model.predict(X)[0]

        logging.info(f"Raw prediction from model: {raw_pred}")

        # If model outputs integer index
        if isinstance(raw_pred, (int, np.integer)):
            if 0 <= raw_pred < len(CLASS_LABELS):
                prediction = CLASS_LABELS[raw_pred]
            else:
                prediction = "Unknown"
        else:
            # If model outputs a string label
            prediction = str(raw_pred)

        # --------------------------
        # Try to get probabilities
        # --------------------------
        top = []
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            top_indices = probs.argsort()[::-1][:3]
            top = [
                {"label": CLASS_LABELS[i], "prob": float(probs[i])}
                for i in top_indices
            ]

        return {
            "prediction": prediction,
            "prediction_index": int(raw_pred) if isinstance(raw_pred, (int, np.integer)) else None,
            "top": top,
        }

    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# -------------------------------------------------------------
# RUN SERVER (local only)
# -------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
