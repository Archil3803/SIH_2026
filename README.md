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
- **Hierarchical Species & Breed Verification**: Two-tiered validation ensuring biological consistency.
- **Veterinary Intelligence Profiles**: Complete database of milk yield, origin, climate adaptation, physical markers, and common health vulnerabilities.
- **Progressive Web App (PWA)**: Installable on Android, iOS, and PC with real-time camera inference support.
- **Offline Resilient**: Local database fallback and service worker caching.
- **RESTful API**: Clean JSON endpoints for mobile and edge device integration.

---

## 📁 Repository Structure

```
├── app.py                     # Flask web server and API endpoints
├── inference.py               # Core PyTorch inference engine & confidence scoring
├── species_detector.py        # Species boundary validation
├── model.py                   # Neural network architecture definitions
├── data_loader.py             # Data loading and augmentation pipelines
├── train.py                   # Model training workflow
├── data/
│   └── breed_database.json    # Comprehensive veterinary & breed metadata
├── models/
│   └── cattle_classifier.pth  # Trained model weights
├── dataset/                   # Curated dataset classes
├── static/                    # Glassmorphism UI styles, PWA manifest, and frontend JS
├── templates/                 # HTML templates
├── Dockerfile                 # Container deployment specification
├── docker-compose.yml         # Container orchestration
├── render.yaml                # Render cloud deployment blueprint
├── requirements.txt           # Python dependencies
└── start_public_tunnel.bat    # Cloudflare public access tunnel script
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
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

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
