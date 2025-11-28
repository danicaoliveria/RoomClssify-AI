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

        # Load as grayscale
        img = Image.open(io.BytesIO(contents)).convert("L")

        # Resize to 32 × 32 → 1024 pixels
        img = img.resize((32, 32))

        # Convert to 1000-feature vector
        arr = np.array(img).flatten().astype(np.float32)
        arr = arr[:1000]

        # Normalize 0–1 (matches dataset creation)
        arr = arr / 255.0

        X = arr.reshape(1, -1)

        # Try to get probabilities
        top = []
        try:
            probs = model.predict_proba(X)[0]
            idx = np.argsort(probs)[::-1][:3]  # top 3 indices
            top = [
                {"label": CLASS_LABELS[i], "prob": float(probs[i])}
                for i in idx
            ]
        except:
            logging.info("Model has no predict_proba")

        # Predict class index
        pred_index = int(model.predict(X)[0])
        logging.info(f"Pred index = {pred_index}")

        if pred_index < 0 or pred_index >= len(CLASS_LABELS):
            return {"prediction": "Unknown", "top": top}

        return {
            "prediction": CLASS_LABELS[pred_index],
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
