# 🧠 NeuroScan XAI — Brain MRI Tumor Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-00e5a0?style=flat-square)

**An explainable AI (XAI) web application for brain MRI tumor classification, powered by ResNet-18 and three interpretability methods: Grad-CAM, SHAP, and LIME.**

### 🌐 [Live Demo → explaiin-your-brain-image-with-xai-1.onrender.com](https://explaiin-your-brain-image-with-xai-1.onrender.com/)

</div>

---

## 📸 Overview

NeuroScan XAI combines deep learning classification with visual explanation tools, letting you see **why** the model made its prediction — not just **what** it predicted. Upload a brain MRI scan, choose an XAI method, and get an annotated saliency map highlighting the regions that influenced the decision.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is not a certified medical device and must not be used for clinical diagnosis or patient care.

---

## ✨ Features

- **Binary classification** — Normal vs. Tumor using a fine-tuned ResNet-18
- **Three XAI methods** — Grad-CAM, SHAP, and LIME, selectable per inference
- **Confidence scores** — Per-class probabilities displayed with animated progress bars
- **Drag-and-drop upload** — Supports JPG, PNG, BMP, WEBP
- **Dark medical-tech UI** — Clean, professional interface built for readability
- **CPU inference** — No GPU required; runs on standard hosting environments
- **REST API** — FastAPI backend with `/predict` endpoint

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.9+ |
| Model | PyTorch, ResNet-18 |
| XAI | Grad-CAM (custom), SHAP (GradientExplainer), LIME (lime-image) |
| Frontend | Jinja2 templates, vanilla HTML/CSS/JS |
| Deployment | Render.com |

---

## 🔍 XAI Methods Explained

### Grad-CAM
Gradient-weighted Class Activation Mapping uses the gradients of the predicted class flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions. **Fast, no extra dependencies beyond PyTorch.**

### SHAP
SHapley Additive exPlanations uses game-theory-based attribution to explain the contribution of each pixel to the prediction. This app uses `shap.GradientExplainer` for efficient deep learning attribution. **More theoretically rigorous**, though slower than Grad-CAM.

### LIME
Local Interpretable Model-agnostic Explanations perturbs the input image into superpixels and fits a local linear model to explain which regions most influenced the prediction. **Model-agnostic** — works with any classifier.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- The trained model file: `best_brain_mri_resnet18.pth`

### 1. Clone the repository

```bash
git clone https://github.com/your-username/neuroscan-xai.git
cd neuroscan-xai
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn torch torchvision pillow matplotlib
pip install shap lime scikit-image        # for SHAP and LIME support
```

Or with a requirements file:

```bash
pip install -r requirements.txt
```

### 3. Add the model weights

Place your trained model file in the project root:

```
neuroscan-xai/
├── app.py
├── best_brain_mri_resnet18.pth   ← place here
├── templates/
│   └── index.html
└── static/
```

The checkpoint must contain:

```python
{
    "model_state_dict": ...,   # ResNet-18 weights
    "class_names": [...],      # e.g. ["Normal", "Tumor"]
    "img_size": 224            # input resolution
}
```

### 4. Run the application

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📁 Project Structure

```
neuroscan-xai/
├── app.py                    # FastAPI application & all inference logic
├── best_brain_mri_resnet18.pth
├── requirements.txt
├── templates/
│   └── index.html            # Jinja2 frontend template
└── static/
    ├── uploads/              # Temporarily stores uploaded MRI images
    └── results/              # Stores generated explanation overlays
```

---

## 🔌 API Reference

### `GET /`
Returns the main HTML interface.

### `POST /predict`
Runs inference and returns the rendered result page.

**Form fields:**

| Field | Type | Values | Description |
|---|---|---|---|
| `file` | file | image/* | MRI image to classify |
| `explanation_method` | string | `gradcam`, `shap`, `lime` | XAI method to use |

**Response context variables:**

| Variable | Type | Description |
|---|---|---|
| `prediction` | string | `"Normal"` or `"Tumor"` |
| `confidence` | float | Confidence % of the predicted class |
| `normal_prob` | float | Probability % for Normal class |
| `tumor_prob` | float | Probability % for Tumor class |
| `uploaded_image` | string | Static URL to the uploaded image |
| `explanation_image` | string | Static URL to the generated saliency map |
| `selected_method` | string | Human-readable method name |

---

## 🧪 Model Details

- **Architecture:** ResNet-18 with a custom fully connected output layer
- **Classes:** Normal, Tumor (binary)
- **Input size:** 224 × 224 RGB
- **Inference device:** CPU (no CUDA required)
- **Grad-CAM target layer:** `model.layer4[-1]` (last ResNet block)

---

## 🌐 Deployment

This app is deployed on **Render.com**.

### Deploy your own instance

1. Push your repo to GitHub (including the `.pth` model file or use Git LFS)
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set the build command:
   ```
   pip install -r requirements.txt
   ```
4. Set the start command:
   ```
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
5. Deploy — your app will be live at your Render URL

> **Note:** Render free-tier instances spin down after inactivity. The first request after a cold start may take 30–60 seconds.

---

## 📦 requirements.txt (recommended)

```txt
fastapi
uvicorn[standard]
torch
torchvision
pillow
matplotlib
numpy
shap
lime
scikit-image
python-multipart
jinja2
aiofiles
```

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/) — deep learning framework
- [SHAP](https://github.com/shap/shap) — model explanation library
- [LIME](https://github.com/marcotcr/lime) — local model interpretability
- [FastAPI](https://fastapi.tiangolo.com/) — modern Python web framework
- Brain MRI dataset sourced from publicly available research repositories

---

<div align="center">
  <sub>Built for education and research. Not for medical use.</sub>
</div>
