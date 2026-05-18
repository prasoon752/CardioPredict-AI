# 🫀 CardioPredict AI

A full-stack cardiovascular disease risk prediction web application powered by a Deep Neural Network with SHAP explainability.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0.3-black?style=flat-square&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?style=flat-square&logo=tensorflow)
![SHAP](https://img.shields.io/badge/SHAP-0.45-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 Overview

CardioPredict AI accepts 14 clinical, lifestyle, and genetic inputs and returns:
- **Real-time CVD risk score** (0–100)
- **Risk level** (Low / Moderate / High / Very High)
- **SHAP-based factor explanations** — which inputs drove the prediction
- **Clinical recommendations** tailored to the patient profile
- **Clinical insight** generated from the model output

> ⚠️ **Disclaimer:** This is an academic research project. Not a substitute for professional medical advice.

---

## 🏗️ Project Structure

```
CardioPredict_AI/
├── backend/
│   ├── app.py              # Flask API (fixed & production ready)
│   ├── requirements.txt    # Python dependencies
│   ├── .env.example        # Environment variable template
│   └── model/
│       └── README.txt      # Place your model files here
├── frontend/
│   └── index.html          # Complete UI (HTML + CSS + JS)
├── render.yaml             # Render.com deployment config
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-CORS |
| ML Model | TensorFlow/Keras (.h5) or scikit-learn (.pkl) |
| Explainability | SHAP (KernelExplainer / TreeExplainer) |
| Frontend | Vanilla JS, HTML5, CSS3 |
| Deployment | Render (backend) + Netlify (frontend) |

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 85% | 0.88 |
| Random Forest | 89% | 0.91 |
| XGBoost | 91% | 0.93 |
| **Deep Neural Network** | **93%** | **0.95** |

---

## 🚀 Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/prasoon752/CardioPredict-AI.git
cd CardioPredict-AI
```

### 2. Set up the backend
```bash
cd backend
pip install -r requirements.txt
```

### 3. Add your model files
Place your trained files inside `backend/model/`:
- `cardio_model.h5` — your Keras model
- `scaler.pkl` — your fitted StandardScaler

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Run the backend
```bash
python app.py
# Backend runs at http://localhost:5000
```

### 6. Open the frontend
Open `frontend/index.html` in your browser.
Make sure `BACKEND_URL` in index.html is set to `http://localhost:5000`.

---

## 🌐 Deployment

### Backend → Render.com
1. Push code to GitHub
2. Connect repo to Render
3. `render.yaml` handles configuration automatically
4. Add `MODEL_PATH` and `SCALER_PATH` as environment variables in Render dashboard

### Frontend → Netlify
1. Drag and drop `frontend/` folder to Netlify
2. Update `BACKEND_URL` in `index.html` to your Render URL

---

## 🔧 Backend Fixes Applied

| # | Issue | Fix |
|---|---|---|
| 1 | `load_assets()` not called on Gunicorn startup | Called at module import level |
| 2 | CORS wildcard in production | Configurable via `ALLOWED_ORIGINS` env var |
| 3 | No input validation on `/predict` | Added field validation with range checks |
| 4 | Probability could exceed \[0,1\] | Clamped to \[0.0, 1.0\] |
| 5 | Raw tracebacks exposed to client | Generic error message returned |
| 6 | `render.yaml` rootDir mismatch | Fixed to point to `backend/` folder |
| 7 | No `.gitignore` | Added — model files excluded from GitHub |
| 8 | No `.env.example` | Added for easy local setup |

---

## 📥 API Reference

### `POST /predict`

**Request Body:**
```json
{
  "age": 55,
  "sex": "M",
  "systolicBP": 145,
  "diastolicBP": 90,
  "cholesterol": 2,
  "bmi": 28.5,
  "glucose": 95,
  "smoking": 1,
  "alcohol": 0,
  "physicalActivity": 0,
  "diabetes": 0,
  "prs": "high"
}
```

**Response:**
```json
{
  "riskScore": 75,
  "riskLevel": "High",
  "probability": 0.7542,
  "shapFactors": [...],
  "recommendations": [...],
  "clinicalInsight": "...",
  "modelUsed": "trained_model",
  "shapUsed": true
}
```

### `GET /health`
Returns server status and uptime.

---

## 👤 Author

**Prasoon Kumar**
B.Tech CSE · Galgotias University

---

## 📄 License

MIT License — free to use for educational and research purposes.
