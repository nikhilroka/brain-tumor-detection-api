import os
from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# =========================================================
# Configuration
# =========================================================

DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "Brain Tumor",
    "Normal"
]


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="Brain Tumor Detection API",
    description="ResNet50-based Brain Tumor Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Image preprocessing
# =========================================================

predict_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# Load ResNet50 model
# =========================================================

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# =========================================================
# Model path
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "brain_tumor_resnet50.pth"
)


# =========================================================
# Load trained weights
# =========================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()


# =========================================================
# Home endpoint
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Brain Tumor Detection API is running",
        "status": "success"
    }


# =========================================================
# Health endpoint
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "ResNet50",
        "device": "CPU"
    }


# =========================================================
# Prediction endpoint
# =========================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # -----------------------------------------------------
    # Allowed image formats
    # -----------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg"
    }

    # -----------------------------------------------------
    # Check file type
    # -----------------------------------------------------

    if file.content_type not in allowed_types:

        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid file type",
                "message": "Please upload a JPG or PNG image."
            }
        )

    try:

        # -------------------------------------------------
        # Read uploaded file
        # -------------------------------------------------

        image_bytes = await file.read()

        # -------------------------------------------------
        # Limit file size to 10 MB
        # -------------------------------------------------

        max_size = 10 * 1024 * 1024

        if len(image_bytes) > max_size:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "File too large",
                    "message": "Maximum file size is 10 MB."
                }
            )

        # -------------------------------------------------
        # Convert bytes to image
        # -------------------------------------------------

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # -------------------------------------------------
        # Preprocess image
        # -------------------------------------------------

        image_tensor = predict_transform(
            image
        )

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(
            DEVICE
        )

        # -------------------------------------------------
        # Model prediction
        # -------------------------------------------------

        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_class = torch.max(
                probabilities,
                dim=1
            )

        # -------------------------------------------------
        # Convert prediction to Python values
        # -------------------------------------------------

        predicted_index = predicted_class.item()

        prediction = str(
            CLASS_NAMES[predicted_index]
        )

        confidence = confidence.item()

        # -------------------------------------------------
        # Professional API response
        # -------------------------------------------------

        return {
            "status": "success",
            "filename": file.filename,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "confidence_percent": round(
                confidence * 100,
                2
            ),
            "model": "ResNet50",
            "model_version": "1.0"
        }

    # -----------------------------------------------------
    # Handle invalid images / prediction errors
    # -----------------------------------------------------

    except Exception as e:

        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid image",
                "message": str(e)
            }
        )

