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

# Serve JS, CSS, images, and components
app.mount("/Components", StaticFiles(directory="Components"), name="components")
app.mount("/images", StaticFiles(directory="images"), name="images")

# Serve root HTML files
@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/train")
def train_page():
    return FileResponse("train.html")

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

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((100, 100))
        arr = np.array(img).flatten().reshape(1, -1)

        prediction = model.predict(arr)[0]
        return {"prediction": str(prediction)}

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
