import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from torchvision import models, transforms


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_brain_mri_resnet18.pth"

STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"
TEMPLATE_DIR = BASE_DIR / "templates"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CPU-only inference
# ============================================================

device = torch.device("cpu")


# ============================================================
# Load model
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Make sure best_brain_mri_resnet18.pth is in the same folder as app.py"
    )

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

class_names = checkpoint.get("class_names", ["Normal", "Tumor"])
num_classes = len(class_names)
IMG_SIZE = checkpoint.get("img_size", 224)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


# ============================================================
# Transforms
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# FastAPI setup
# ============================================================

app = FastAPI(title="Brain MRI XAI Classifier")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# ============================================================
# Prediction
# ============================================================

def predict_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)

    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

    return image_tensor, pred_idx, probs.cpu().numpy()


def get_probability(class_label, probs):
    if class_label in class_names:
        return round(float(probs[class_names.index(class_label)]) * 100, 2)
    return None


# ============================================================
# Grad-CAM
# ============================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx):
        self.model.eval()

        input_tensor = image_tensor.unsqueeze(0).to(device)
        logits = self.model(input_tensor)

        self.model.zero_grad()
        logits[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, weight in enumerate(weights):
            cam += weight.cpu() * activations[i].cpu()

        cam = torch.relu(cam).numpy()

        cam_img = Image.fromarray(cam)
        cam_img = cam_img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
        cam = np.array(cam_img)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


gradcam = GradCAM(model, model.layer4[-1])


def save_gradcam_overlay(image_tensor, class_idx, save_path: Path):
    cam = gradcam.generate(image_tensor, class_idx)

    image_np = image_tensor.permute(1, 2, 0).numpy()
    heatmap = plt.cm.jet(cam)[..., :3]
    overlay = 0.45 * heatmap + 0.55 * image_np
    overlay = np.clip(overlay, 0, 1)

    save_figure(image_np, overlay, "Original MRI", "Grad-CAM Explanation", save_path)


# ============================================================
# SHAP
# ============================================================

def save_shap_explanation(image_tensor, class_idx, save_path: Path):
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is not installed. Run: pip install shap")

    model.eval()

    # For a simple web demo, use the uploaded image as a small background reference.
    # This keeps CPU inference lightweight.
    background = image_tensor.unsqueeze(0).to(device)
    sample = image_tensor.unsqueeze(0).to(device)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        sv = np.array(shap_values[class_idx][0])
    else:
        shap_arr = np.array(shap_values)

        if shap_arr.ndim == 5 and shap_arr.shape[-1] == num_classes:
            sv = shap_arr[0, :, :, :, class_idx]
        elif shap_arr.ndim == 5 and shap_arr.shape[1] == num_classes:
            sv = shap_arr[0, class_idx]
        else:
            sv = shap_arr[0]

    if sv.ndim == 3 and sv.shape[0] == 3:
        sv = np.transpose(sv, (1, 2, 0))

    if sv.ndim == 3:
        shap_heatmap = np.mean(np.abs(sv), axis=-1)
    else:
        shap_heatmap = np.abs(sv)

    shap_heatmap = shap_heatmap - shap_heatmap.min()
    shap_heatmap = shap_heatmap / (shap_heatmap.max() + 1e-8)

    image_np = image_tensor.permute(1, 2, 0).numpy()

    heatmap = plt.cm.hot(shap_heatmap)[..., :3]
    overlay = 0.50 * heatmap + 0.50 * image_np
    overlay = np.clip(overlay, 0, 1)

    save_figure(image_np, overlay, "Original MRI", "SHAP Explanation", save_path)


# ============================================================
# LIME
# ============================================================

def lime_predict_fn(images_np):
    model.eval()

    batch = torch.tensor(images_np, dtype=torch.float32)
    batch = batch.permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)

    return probs.cpu().numpy()


def save_lime_explanation(image_tensor, class_idx, save_path: Path):
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        raise ImportError("LIME or scikit-image is not installed. Run: pip install lime scikit-image")

    image_np = image_tensor.permute(1, 2, 0).numpy()

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image_np,
        lime_predict_fn,
        top_labels=num_classes,
        hide_color=0,
        num_samples=500,
    )

    temp, mask = explanation.get_image_and_mask(
        label=class_idx,
        positive_only=True,
        num_features=8,
        hide_rest=False,
    )

    lime_overlay = mark_boundaries(temp, mask)

    save_figure(image_np, lime_overlay, "Original MRI", "LIME Explanation", save_path)


# ============================================================
# Plot helper
# ============================================================

def save_figure(original, explanation, left_title, right_title, save_path: Path):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title(left_title)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(explanation)
    plt.title(right_title)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def generate_explanation(method, image_tensor, pred_idx, result_path):
    method = method.lower()

    if method == "gradcam":
        save_gradcam_overlay(image_tensor, pred_idx, result_path)
        return "Grad-CAM"

    if method == "shap":
        save_shap_explanation(image_tensor, pred_idx, result_path)
        return "SHAP"

    if method == "lime":
        save_lime_explanation(image_tensor, pred_idx, result_path)
        return "LIME"

    save_gradcam_overlay(image_tensor, pred_idx, result_path)
    return "Grad-CAM"


# ============================================================
# Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    explanation_method: str = Form("gradcam")
):
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"error": "Please upload a valid image file: jpg, jpeg, png, bmp, or webp."}
        )

    unique_id = str(uuid.uuid4())
    unique_name = f"{unique_id}{file_ext}"
    upload_path = UPLOAD_DIR / unique_name

    file_bytes = await file.read()

    with open(upload_path, "wb") as f:
        f.write(file_bytes)

    try:
        image_tensor, pred_idx, probs = predict_image(upload_path)

        result_name = f"{explanation_method}_{unique_id}.png"
        result_path = RESULT_DIR / result_name

        selected_method = generate_explanation(
            explanation_method,
            image_tensor,
            pred_idx,
            result_path,
        )

        prediction = class_names[pred_idx]
        confidence = round(float(probs[pred_idx]) * 100, 2)

        normal_prob = get_probability("Normal", probs)
        tumor_prob = get_probability("Tumor", probs)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": prediction,
                "confidence": confidence,
                "normal_prob": normal_prob,
                "tumor_prob": tumor_prob,
                "uploaded_image": f"/static/uploads/{unique_name}",
                "explanation_image": f"/static/results/{result_name}",
                "selected_method": selected_method,
                "selected_value": explanation_method,
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": f"Analysis failed: {str(e)}",
                "selected_value": explanation_method,
            }
        )
