# Bovista - Veterinary Intelligence & Bovine Breed Classifier 🐄🐃
> **Smart India Hackathon (SIH 2026)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bovista is an AI-powered veterinary diagnostic and bovine classification platform capable of identifying indigenous and exotic cattle and buffalo breeds in real-time with comprehensive physical, biological, and veterinary profiles.

---

## 🌟 Key Features

- **High-Precision Multi-Class Classification**: AI classification for cattle and buffalo breeds.
- **Decoupled Frontend & Backend**: Clean, modular structure supporting both integrated serving and decoupled frontend client development.
- **Hierarchical Species & Breed Verification**: Two-tiered validation ensuring biological consistency.
- **Veterinary Intelligence Profiles**: Complete database of milk yield, origin, climate adaptation, physical markers, and common health vulnerabilities.
- **Progressive Web App (PWA)**: Installable on Android, iOS, and PC with real-time camera inference support.
- **Offline Resilient**: Local database fallback and service worker caching.
- **RESTful API**: Clean JSON endpoints with CORS enabled for mobile and edge device integration.

---

## 📁 Repository Structure

```
SIH_2026/
├── frontend/                     # Pure Frontend Web App & PWA Client
│   ├── index.html                # Single-page application dashboard
│   ├── css/
│   │   └── style.css             # Glassmorphism design system & animations
│   ├── js/
│   │   ├── app.js                # Dynamic client logic (supports standalone & API modes)
│   │   └── qrcode.min.js         # Mobile browser QR generator
│   ├── images/                   # Icons, logos, and UI imagery
│   ├── manifest.json             # PWA web manifest
│   ├── sw.js                     # Offline caching service worker
│   ├── BovineAI-Mobile-v1.0.apk  # Android mobile APK package
│   └── README.md                 # Frontend documentation
│
├── backend/                      # Python / PyTorch Backend API Service
│   ├── app.py                    # Flask REST API server and static frontend bridge
│   ├── inference.py              # PyTorch inference engine & multi-bovine detection
│   ├── species_detector.py       # Species boundary validation
│   ├── bovine_detector.py        # Bovine localization & bounding box segmentation
│   ├── model.py                  # Neural network architecture definitions
│   ├── data_loader.py            # Supervised dataset loader & augmentation pipeline
│   ├── train.py                  # Model training and fine-tuning workflow
│   ├── predict_cli.py            # CLI prediction tool
│   ├── requirements.txt          # Python dependencies
│   ├── data/
│   │   ├── breed_database.json   # 10-breed veterinary knowledge base
│   │   └── dataset.csv           # Supervised dataset catalog
│   ├── models/
│   │   ├── cattle_classifier.pth # Trained model weights
│   │   └── classes.json          # Standardized class mappings
│   ├── tests/                    # Unit, K-Fold, and API verification test suite
│   │   ├── test_system.py
│   │   ├── test_kfold_integrity.py
│   │   ├── test_all_classes.py
│   │   └── verify_all_endpoints.py
│   └── README.md                 # Backend API documentation
│
├── dataset/                      # Curated image dataset (shared/referenced)
├── run.py                        # Root launcher script
├── app.py                        # Root backward-compatibility bridge
├── requirements.txt              # Root Python dependencies
├── Dockerfile                    # Container deployment specification
├── docker-compose.yml            # Container orchestration
├── render.yaml                   # Render cloud deployment blueprint
├── Procfile                      # WSGI production runner
├── start_public_tunnel.bat       # Cloudflare public access tunnel script
└── DEPLOYMENT.md                 # Deployment guides
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- pip

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Archil3803/SIH_2026.git
cd SIH_2026

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Locally

#### Integrated Full-Stack Mode (Single Command)
```bash
python run.py
# or
python backend/app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

#### Decoupled Frontend Development
```bash
# In terminal 1: Run Backend API
python backend/app.py

# In terminal 2: Run Frontend Static Server
cd frontend
python -m http.server 3000
```
Open [http://localhost:3000](http://localhost:3000). The frontend automatically connects to the backend API via CORS.

---

## 🧪 Testing

```bash
# Run unit & system tests
python backend/tests/test_system.py

# Run K-Fold integrity audit
python backend/tests/test_kfold_integrity.py

# Run end-to-end API verification (with server running)
python backend/tests/verify_all_endpoints.py
```

---

## ☁️ Deployment

Refer to [`DEPLOYMENT.md`](DEPLOYMENT.md) for step-by-step guides to deploy on:
- **Render.com** (Auto-deploy with `render.yaml`)
- **Docker / Docker Compose**
- **Hugging Face Spaces**
- **Cloudflare Tunnels**

---

## 📄 License
This project is licensed under the MIT License.
