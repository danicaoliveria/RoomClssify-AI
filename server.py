# server.py
import io
import os
import pickle
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import uvicorn
import torch
from torchvision import models, transforms

# ---------------------------------------------------
# FASTAPI + CORS
# ---------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------
CLASS_LABELS = [
    "Bathroom", "Bedroom", "Dining", "Gaming", "Kitchen",
    "Laundry", "Living", "Office", "Terrace", "Yard"
]

# ---------------------------------------------------
# LOAD SQUEEZENET (extract 1000 features)
# ---------------------------------------------------
device = torch.device("cpu")

snet = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
snet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------------------------
# LOAD SKLEARN MODEL
# ---------------------------------------------------
MODEL_PATH = "model/model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Loaded sklearn model!")
except Exception as e:
    print("❌ Failed loading model:")
    print(e)
    model = None

# ---------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------
def extract_features(img: Image.Image):
    img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = snet(tensor)  # [1, 1000]

    features = logits.cpu().numpy().astype(np.float32)

    # sklearn expects shape (1, 1000)
    return features

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Load image
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
    except:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Extract features
    try:
        feat = extract_features(img)
        print("🔍 Feature shape =", feat.shape)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Feature extraction error: {e}")

    # Predict
    try:
        pred_index = int(model.predict(feat)[0])
        room_label = CLASS_LABELS[pred_index]
        print("🎯 Prediction:", room_label)
        return {"prediction": room_label}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
