import os
import io
import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import uvicorn

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "room_classifier.pkcls")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model file not found on server")

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print("❌ ERROR loading model:", e)
        raise RuntimeError("Failed to load model")

model = load_model()

# ---------------------------------------------------
# FASTAPI SETUP
# ---------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve your frontend
app.mount("/Components", StaticFiles(directory=os.path.join(BASE_DIR, "Components")), name="Components")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

# ---------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------
def preprocess_image(file):
    img = Image.open(io.BytesIO(file))
    img = img.convert("RGB")
    img = img.resize((100, 100))
    arr = np.array(img).flatten().reshape(1, -1)
    return arr

# ---------------------------------------------------
# PREDICT ENDPOINT
# ---------------------------------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        processed = preprocess_image(contents)
        prediction = model.predict(processed)[0]

        return {"prediction": str(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------
# RUN SERVER (Render auto-runs with Uvicorn)
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
