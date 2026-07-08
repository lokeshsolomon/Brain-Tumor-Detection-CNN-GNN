"""
app.py
======
Flask web application for the Brain Tumor Detection & Classification project.

- Serves an upload form (templates/index.html)
- Accepts a brain MRI image
- Runs the trained CNN+GAT model (model.py) to predict:
    * Tumor presence
    * Tumor type / class
    * Confidence score (per-class probabilities)
- Generates a Grad-CAM heatmap overlay highlighting the predicted region
- Renders the result (templates/result.html)

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash
import matplotlib.cm as mpl_cm

from model import (
    BrainTumorDetector, GradCAM, get_transforms,
    CLASS_NAMES, CLASS_INFO, IMG_SIZE,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pth")
UPLOAD_DIR  = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "brain-tumor-cnn-gat-demo"  # only used for flash messages
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_model = None
_gradcam = None


def get_model():
    """Lazily load the model + Grad-CAM hook (only once)."""
    global _model, _gradcam
    if _model is None:
        _model = BrainTumorDetector(pretrained=False).to(DEVICE)
        if os.path.exists(MODEL_PATH):
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
            state_dict = ckpt.get("model_state_dict", ckpt)
            _model.load_state_dict(state_dict)
        else:
            # No trained checkpoint yet -- the app will still run so the UI
            # can be previewed, but predictions will be meaningless until
            # you train the model (see brain_tumor_cnn_gnn.ipynb) and place
            # the checkpoint at models/best_model.pth
            print(f"[WARNING] No checkpoint found at {MODEL_PATH}. "
                  f"Predictions will use randomly-initialized weights.")
        _model.eval()
        _gradcam = GradCAM(_model)
    return _model, _gradcam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def run_inference(image_path: str):
    """
    Run the model + Grad-CAM on an image file.
    Returns a dict with prediction info and the path to the saved overlay image.
    """
    model, gradcam = get_model()

    transform = get_transforms("val")
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    # Prediction + Grad-CAM (single forward+backward pass)
    cam, pred_idx, probs = gradcam(x)

    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    info = CLASS_INFO[pred_class]

    # Build the Grad-CAM overlay image
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_resized) / 255.0

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
    cam_resized = np.array(cam_img) / 255.0
    heatmap = mpl_cm.jet(cam_resized)[..., :3]

    alpha = 0.45
    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    overlay_name = f"gradcam_{uuid.uuid4().hex}.png"
    overlay_path = os.path.join(UPLOAD_DIR, overlay_name)
    Image.fromarray(overlay).save(overlay_path)

    class_probs = [
        {"name": CLASS_INFO[c]["label"], "value": float(p)}
        for c, p in zip(CLASS_NAMES, probs)
    ]
    class_probs.sort(key=lambda d: d["value"], reverse=True)

    return {
        "tumor_present": info["tumor_present"],
        "predicted_class": info["label"],
        "description": info["description"],
        "confidence": confidence,
        "class_probs": class_probs,
        "gradcam_filename": overlay_name,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file selected.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PNG or JPG image.")
        return redirect(url_for("index"))

    upload_name = f"upload_{uuid.uuid4().hex}_{file.filename}"
    upload_path = os.path.join(UPLOAD_DIR, upload_name)
    file.save(upload_path)

    try:
        result = run_inference(upload_path)
    except Exception as exc:
        flash(f"Error during prediction: {exc}")
        return redirect(url_for("index"))

    return render_template(
        "result.html",
        original_image=f"uploads/{upload_name}",
        gradcam_image=f"uploads/{result['gradcam_filename']}",
        tumor_present=result["tumor_present"],
        predicted_class=result["predicted_class"],
        description=result["description"],
        confidence=round(result["confidence"] * 100, 2),
        class_probs=[
            {"name": p["name"], "value": round(p["value"] * 100, 2)}
            for p in result["class_probs"]
        ],
    )


if __name__ == "__main__":
    # Pre-load the model on startup so the first request isn't slow
    get_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
