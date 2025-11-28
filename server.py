import os
import io
import numpy as np
import Orange
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
MODEL_PATH = os.path.join(BASE_DIR, "model", "roomclassify.pkcls")

# ---------------------------------------------------
# LOAD ORANGE MODEL
# ---------------------------------------------------
def load_model():
    print(f"🔍 Checking model file: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found!")
        raise RuntimeError("Model file not found on server")

    try:
        # CORRECT WAY to load Orange .pkcls model
        model = Orange.classification.Model.from_pickle(MODEL_PATH)
        print("✅ Orange model loaded successfully")
        return model

    except Exception as e:
        print("❌ Error loading Orange model:", e)
        raise RuntimeError("Failed to load Orange model")


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

# Serve folder for frontend files
app.mount("/Components", StaticFiles(directory=os.path.join(BASE_DIR, "Components")), name="Components")
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

# ---------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------
def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("RGB")
    img = img.resize((100, 100))

    # Flatten image into numeric list
    arr = np.array(img).flatten()
    return arr.tolist()

# ---------------------------------------------------
# PREDICTION ENDPOINT
# ---------------------------------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        feature_list = preprocess_image(contents)

        # Convert list → Orange Instance
        example = Orange.data.Instance(model.domain, feature_list)

        # Perform prediction
        prediction = model(example)

        return {"prediction": str(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# ---------------------------------------------------
# UVICORN RUNNER (RENDER USES THIS)
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
