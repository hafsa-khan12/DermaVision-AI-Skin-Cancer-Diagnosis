import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transfer_resnet import TransferResNet


APP_NAME = "DermaVision AI"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "checkpoints"
    / "transfer_full_finetune_best.pth"
)

UPLOAD_DIR = PROJECT_ROOT / "static" / "uploads"
PREPROCESS_DIR = PROJECT_ROOT / "static" / "preprocessed"
LOG_PATH = PROJECT_ROOT / "reports" / "final_results" / "app_prediction_log.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_NAME)

app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

MODEL = None
CLASS_NAMES = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_once():
    """Load trained model once at startup."""
    global MODEL, CLASS_NAMES

    if MODEL is not None:
        return MODEL, CLASS_NAMES

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    CLASS_NAMES = checkpoint["class_names"]

    MODEL = TransferResNet(
        num_classes=len(CLASS_NAMES),
        dropout=0.3,
        freeze_backbone=False,
    ).to(DEVICE)

    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.eval()

    return MODEL, CLASS_NAMES


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Use same preprocessing style as training."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = image.convert("RGB")
    return transform(image).unsqueeze(0)


def create_preprocessing_previews(pil_image: Image.Image, safe_name: str):
    """
    Save resized and normalized preview images for the UI.
    These images help show the live preprocessing pipeline visually.
    """

    # Step 1: resized preview image
    resized_preview = pil_image.resize((224, 224))
    resized_name = f"resized_{safe_name}"
    resized_path = PREPROCESS_DIR / resized_name
    resized_preview.save(resized_path)

    # Step 2: normalized preview image for visual display
    normalize_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    normalized_tensor = normalize_transform(pil_image)

    # Convert normalized tensor back into a viewable image
    preview_tensor = normalized_tensor.clone()
    preview_tensor = preview_tensor.permute(1, 2, 0)
    preview_tensor = preview_tensor - preview_tensor.min()

    if preview_tensor.max() > 0:
        preview_tensor = preview_tensor / preview_tensor.max()

    preview_array = (preview_tensor.numpy() * 255).astype("uint8")

    normalized_image = Image.fromarray(preview_array)
    normalized_name = f"normalized_{safe_name}"
    normalized_path = PREPROCESS_DIR / normalized_name
    normalized_image.save(normalized_path)

    return resized_name, normalized_name


def risk_from_prediction(prediction: str, confidence: float) -> str:
    """Generate risk label from prediction and confidence."""
    prediction = prediction.lower()

    if prediction == "malignant":
        if confidence >= 0.80:
            return "High Risk"
        return "Medium Risk"

    if confidence >= 0.80:
        return "Low Risk"

    return "Review Recommended"


def generate_summary(prediction: str, confidence: float) -> str:
    """Generate readable AI summary."""
    percent = confidence * 100

    if prediction.lower() == "malignant":
        return (
            f"The AI model analyzed the uploaded skin lesion image and predicted it "
            f"as malignant with a confidence score of {percent:.1f}%. The decision "
            f"is mainly influenced by visual patterns such as irregular lesion "
            f"appearance, color variation, border irregularity, and texture-based "
            f"features learned during training. This result should be used as an "
            f"early screening support tool only, and clinical confirmation by a "
            f"dermatologist is strongly recommended."
        )

    return (
        f"The AI model analyzed the uploaded skin lesion image and predicted it "
        f"as benign with a confidence score of {percent:.1f}%. The image appears "
        f"closer to benign examples learned by the model. However, any lesion that "
        f"changes in size, color, shape, or causes symptoms should still be reviewed "
        f"by a qualified dermatologist."
    )


def log_prediction(filename: str, prediction: str, confidence: float, latency: float) -> None:
    """Save prediction log."""
    file_exists = LOG_PATH.exists()

    with open(LOG_PATH, "a", encoding="utf-8") as file:
        if not file_exists:
            file.write("timestamp,filename,prediction,confidence,latency_seconds\n")

        file.write(
            f"{datetime.now()},{filename},{prediction},{confidence:.4f},{latency:.4f}\n"
        )


def default_scan_context():
    """Default context for scan page before prediction."""
    return {
        "page_title": "New Scan - DermaVision AI",
        "active_page": "scans",
        "heading": "New Skin Lesion",
        "heading_span": "Scan",
        "subtitle": "Upload a lesion image and generate AI diagnosis result.",
        "has_result": False,
        "prediction": "Waiting for Scan",
        "confidence": 0,
        "risk": "Not Available",
        "latency": 0,
        "image_url": "/static/samples/sample_lesion_placeholder.png",
        "resized_image_url": "/static/samples/sample_lesion_placeholder.png",
        "normalized_image_url": "/static/samples/sample_lesion_placeholder.png",
        "summary": "Upload a skin lesion image and run AI diagnosis to generate a medical summary.",
        "probs": [
            ("Malignant", 0),
            ("Benign", 0),
            ("Uncertain", 0),
            ("Needs Review", 0),
            ("Other", 0),
        ],
    }


@app.on_event("startup")
def startup_event():
    """Load model when app starts."""
    try:
        load_model_once()
        print("Model loaded successfully.")
    except Exception as error:
        print("Model loading warning:", error)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "checkpoint_path": str(CHECKPOINT_PATH),
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Dashboard overview page."""
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "page_title": "DermaVision AI Dashboard",
            "active_page": "dashboard",
            "heading": "DermaVision",
            "heading_span": "AI",
            "subtitle": "Early Skin Cancer Diagnosis & Risk Screening",
        },
    )


@app.get("/scans", response_class=HTMLResponse)
def scans_page(request: Request):
    """AI scan upload page."""
    try:
        return templates.TemplateResponse(
            request=request,
            name="scans.html",
            context=default_scan_context(),
        )
    except Exception as error:
        return PlainTextResponse(
            content=f"SCANS PAGE ERROR:\n\n{type(error).__name__}\n\n{str(error)}",
            status_code=500,
        )


@app.get("/patients", response_class=HTMLResponse)
def patients_page(request: Request):
    """Patient records page."""
    return templates.TemplateResponse(
        request=request,
        name="patients.html",
        context={
            "page_title": "Patients - DermaVision AI",
            "active_page": "patients",
            "heading": "Patient",
            "heading_span": "Records",
            "subtitle": "Demo patient records page for Phase 6 dashboard navigation.",
        },
    )


@app.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request):
    """Analytics page."""
    return templates.TemplateResponse(
        request=request,
        name="analytics.html",
        context={
            "page_title": "Analytics - DermaVision AI",
            "active_page": "analytics",
            "heading": "Diagnosis",
            "heading_span": "Analytics",
            "subtitle": "Model analytics and diagnosis result overview.",
        },
    )


@app.get("/model", response_class=HTMLResponse)
def model_page(request: Request):
    """Model performance page."""
    return templates.TemplateResponse(
        request=request,
        name="model.html",
        context={
            "page_title": "Model Performance - DermaVision AI",
            "active_page": "model",
            "heading": "Model",
            "heading_span": "Performance",
            "subtitle": "Final evaluation metrics, efficiency, and system limitations.",
        },
    )


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    """Settings page."""
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "page_title": "Settings - DermaVision AI",
            "active_page": "settings",
            "heading": "System",
            "heading_span": "Settings",
            "subtitle": "Configuration and deployment information.",
        },
    )


@app.get("/help", response_class=HTMLResponse)
def help_page(request: Request):
    """Help page."""
    return templates.TemplateResponse(
        request=request,
        name="help.html",
        context={
            "page_title": "Help - DermaVision AI",
            "active_page": "help",
            "heading": "Help &",
            "heading_span": "Support",
            "subtitle": "How to use the AI diagnosis dashboard correctly.",
        },
    )


@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    """About project page."""
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={
            "page_title": "About - DermaVision AI",
            "active_page": "about",
            "heading": "About",
            "heading_span": "Project",
            "subtitle": "Project objective, methodology, and educational medical disclaimer.",
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    """Handle upload and prediction."""
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}

    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, and PNG image files are supported.",
        )

    safe_name = f"{int(time.time())}_{image.filename.replace(' ', '_')}"
    save_path = UPLOAD_DIR / safe_name

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        pil_image = Image.open(save_path).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file could not be opened as a valid image.",
        )

    # Create preprocessing preview images for UI
    resized_name, normalized_name = create_preprocessing_previews(
        pil_image=pil_image,
        safe_name=safe_name,
    )

    model, class_names = load_model_once()
    image_tensor = preprocess_image(pil_image).to(DEVICE)

    start_time = time.time()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    latency = time.time() - start_time

    prediction = class_names[predicted_idx.item()]
    confidence_value = float(confidence.item())
    confidence_percent = confidence_value * 100

    display_prediction = prediction.title()
    risk = risk_from_prediction(prediction, confidence_value)
    summary = generate_summary(prediction, confidence_value)

    log_prediction(
        filename=image.filename,
        prediction=display_prediction,
        confidence=confidence_value,
        latency=latency,
    )

    probs = []
    for class_name, prob in zip(class_names, probabilities.cpu().tolist()):
        probs.append((class_name.title(), prob * 100))

    while len(probs) < 5:
        probs.append(("Other", 0.0))

    return templates.TemplateResponse(
        request=request,
        name="scans.html",
        context={
            "page_title": "Diagnosis Result - DermaVision AI",
            "active_page": "scans",
            "heading": "AI Diagnosis",
            "heading_span": "Result",
            "subtitle": "Prediction, confidence score, risk level, and generated report.",
            "has_result": True,
            "prediction": display_prediction,
            "confidence": confidence_percent,
            "risk": risk,
            "latency": latency,
            "image_url": f"/static/uploads/{safe_name}",
            "resized_image_url": f"/static/preprocessed/{resized_name}",
            "normalized_image_url": f"/static/preprocessed/{normalized_name}",
            "summary": summary,
            "probs": probs[:5],
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )