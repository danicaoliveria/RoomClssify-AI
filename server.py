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

# ---------------------------------------------------------
# BASIC CONFIG
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "roomclassify.pkcls")

app = FastAPI()

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# STATIC FILES
# ---------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/Components", StaticFiles(directory="Components"), name="components")

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/train")
def train_page():
    return FileResponse("train.html")

# ---------------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# LOAD ORANGE MODEL
# ---------------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    logging.info("Model loaded: %s", type(model))
    return model

model = load_model()

# Detect expected features
def get_expected_features(model):
    n = getattr(model, "n_features_in_", None)
    if n: 
        return int(n)

    # Pipeline case
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            last = model.steps[-1][1]
            n = getattr(last, "n_features_in_", None)
            if n:
                return int(n)
    except:
        pass

    # Fallback: 1024 (32×32)
    return 32 * 32

EXPECTED_FEATURES = get_expected_features(model)
logging.info(f"Model expects {EXPECTED_FEATURES} input features")

# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale
        image = image.resize((32, 32))

        arr = np.array(image).astype("float32").flatten() / 255.0
        input_len = len(arr)

        # Resize input vector to match training
        if input_len < EXPECTED_FEATURES:
            padded = np.zeros(EXPECTED_FEATURES, dtype="float32")
            padded[:input_len] = arr
            arr = padded
        elif input_len > EXPECTED_FEATURES:
            arr = arr[:EXPECTED_FEATURES]

        X = arr.reshape(1, -1)

        # Predict
        raw_pred = model.predict(X)[0]

        # Convert numeric → label
        if isinstance(raw_pred, (int, np.integer)):
            index = int(raw_pred)
            prediction = CLASS_LABELS[index] if 0 <= index < len(CLASS_LABELS) else "Unknown"
        else:
            prediction = str(raw_pred)

        return {"prediction": prediction}

    except Exception as e:
        logging.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
