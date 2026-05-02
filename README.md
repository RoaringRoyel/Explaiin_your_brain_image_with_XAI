# Brain MRI XAI FastAPI App

This version lets the user upload an MRI image and choose one explanation method:

- Grad-CAM
- SHAP
- LIME

## Required model file

Place your trained model beside `app.py`:

```text
best_brain_mri_resnet18.pth
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Note

This project is educational only and not for medical diagnosis.
