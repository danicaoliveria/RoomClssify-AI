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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static folders
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/Components", StaticFiles(directory="Components"), name="components")

# HTML pages
@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/train")
def train_page():
    return FileResponse("train.html")

# CLASS LABELS (in exact order)
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

# Load model
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found at: " + MODEL_PATH)

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Load image and resize to 100x100
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((100, 100))

        # Flatten image → 100x100x3 = 30,000 features
        arr = np.array(img).flatten().reshape(1, -1)

        # Predict using Orange model
        pred_index = int(model.predict(arr)[0])

        # Convert index → label
        if pred_index < 0 or pred_index >= len(CLASS_LABELS):
            return {"prediction": "Unknown"}

        return {"prediction": CLASS_LABELS[pred_index]}

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

# Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
