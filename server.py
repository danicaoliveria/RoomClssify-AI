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

        # Load image (grayscale)
        img = Image.open(io.BytesIO(contents)).convert("L")

        # Resize to 32×32 (1024 pixels)
        img = img.resize((32, 32))

        # Flatten to 1024
        arr = np.array(img).flatten().astype(np.float32)

        # --- FIX: match model's expected length (1000) ---
        expected_len = 1000
        if arr.size > expected_len:
            arr = arr[:expected_len]
        elif arr.size < expected_len:
            padded = np.zeros(expected_len, dtype=np.float32)
            padded[:arr.size] = arr
            arr = padded
        # ------------------------------------------------

        # Normalize
        arr = arr / 255.0
        X = arr.reshape(1, -1)

        # Predict class index
        pred_index = int(model.predict(X)[0])

        # Top-3 (if available)
        top = []
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            idx = np.argsort(probs)[::-1][:3]
            top = [{
                "label": CLASS_LABELS[i],
                "prob": float(probs[i])
            } for i in idx]

        # Validate prediction
        if pred_index < 0 or pred_index >= len(CLASS_LABELS):
            prediction = "Unknown"
        else:
            prediction = CLASS_LABELS[pred_index]

        return {"prediction": prediction, "top": top}

    except Exception as e:
        logging.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")



# -------------------------------------------------------------
# RUN SERVER (local only)
# -------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
