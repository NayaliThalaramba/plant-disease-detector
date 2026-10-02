import io
import base64
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib
matplotlib.use("Agg")  
from PIL import Image as PILImage

from model import get_device, build_model
from dataset import get_transforms, IMAGENET_MEAN, IMAGENET_STD

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pt")

app = FastAPI(title="Plant Disease Detector API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


device = get_device()
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
CLASS_NAMES = checkpoint["class_names"]
model = build_model(num_classes=len(CLASS_NAMES), freeze_base=False)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

_, eval_transform = get_transforms()
cam_extractor = GradCAM(model=model, target_layers=[model.features[-1]])

print(f"Model loaded on {device}. {len(CLASS_NAMES)} classes available.")


class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    top3: list[dict]
    gradcam_image_base64: str
    low_confidence_warning: bool
    warning_message: str | None = None



LOW_CONFIDENCE_THRESHOLD = 0.70


def unnormalize_for_display(tensor_img):
    img = tensor_img.cpu().numpy().transpose((1, 2, 0))
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = std * img + mean
    return np.clip(img, 0, 1)


def image_array_to_base64(img_array):
    
    pil_img = PILImage.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def format_class_name(raw_name):
    
    parts = raw_name.split("___")
    plant = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else ""
    return f"{plant} - {condition}" if condition else plant


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Plant Disease Detector API is running", "device": str(device)}


@app.get("/classes")
def list_classes():
    return {"num_classes": len(CLASS_NAMES), "classes": [format_class_name(c) for c in CLASS_NAMES]}


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image. Please upload a valid JPG or PNG.")

    image_tensor = eval_transform(image).unsqueeze(0).to(device)

    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top3_probs, top3_indices = torch.topk(probabilities, k=3)
    top3 = [
        {"class_name": format_class_name(CLASS_NAMES[idx.item()]), "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top3_probs, top3_indices)
    ]

    pred_idx = top3_indices[0].item()
    pred_class = format_class_name(CLASS_NAMES[pred_idx])
    pred_confidence = round(top3_probs[0].item(), 4)

    
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam_extractor(input_tensor=image_tensor, targets=targets)[0]

    rgb_img = unnormalize_for_display(image_tensor[0])
    cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)  # uint8 array

    gradcam_b64 = image_array_to_base64(cam_overlay)

    is_low_confidence = pred_confidence <= LOW_CONFIDENCE_THRESHOLD
    warning_message = (
        "The model isn't very confident about this one — it might not be a "
        "clear photo of a leaf, or the disease isn't one of the 38 categories "
        "it was trained on. Treat this result with caution."
        if is_low_confidence else None
    )

    return PredictionResult(
        predicted_class=pred_class,
        confidence=pred_confidence,
        low_confidence_warning=is_low_confidence,
        warning_message=warning_message,
        top3=top3,
        gradcam_image_base64=gradcam_b64,
    )