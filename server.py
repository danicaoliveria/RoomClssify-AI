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
import torch
from torchvision import models, transforms
import traceback


# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

app = FastAPI()

# ---------------------------------------------------
# CORS
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# STATIC FILES
# ---------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/Components", StaticFiles(directory="Components"), name="components")


# ---------------------------------------------------
# HTML ROUTES
# ---------------------------------------------------
@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/train")
def train_page():
    return FileResponse("train.html")


# ---------------------------------------------------
# from fastapi import FastAPI, File, UploadFile, HTTPException
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

# Serve static folders
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/Components", StaticFiles(directory="Components"), name="components")

# Serve root HTML files
@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/train")
def train_page():
    return FileResponse("train.html")


# ---------------------------------------------------
# CLASS LABELS (IN ORDER)
# ---------------------------------------------------
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

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((100, 100))
        arr = np.array(img).flatten().reshape(1, -1)

        pred_index = model.predict(arr)[0]

        # Convert number → class label
        label = CLASS_LABELS[int(pred_index)]

        return {
            "prediction_index": int(pred_index),
            "prediction_label": label
        }

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


# ---------------------------------------------------
# LOAD SQUEEZENET
# ---------------------------------------------------
device = torch.device("cpu")

squeezenet = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
squeezenet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ---------------------------------------------------
# LOAD ORANGE MODEL (.pkl)
# ---------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found at: " + MODEL_PATH)

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()


# ---------------------------------------------------
# FEATURE EXTRACTION FROM IMAGE
# ---------------------------------------------------
def extract_features(img: Image.Image):
    img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = squeezenet(tensor)

    features = outputs.view(outputs.size(0), -1).cpu().numpy()
    return features.astype(np.float32)


# ---------------------------------------------------
# PREDICTION ENDPOINT
# ---------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
    except:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Extract features
    try:
        features = extract_features(img)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Feature extraction error: {e}")

    # Predict class index
    try:
        pred_index = int(model(features)[0])
        label = CLASS_LABELS[pred_index]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"prediction": label}


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
