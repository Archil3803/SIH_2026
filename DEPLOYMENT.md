# Bovista Deployment & Global Access Guide

This document explains how to access and deploy **Bovista Veterinary Intelligence & Bovine Breed Classifier** on any device worldwide.

---

## 🚀 1. Live Instant Public Access (Active Now)

A secure, encrypted Cloudflare HTTPS tunnel is currently running for this application. You can open this link from **any smartphone, tablet, or PC worldwide** (on cellular data, Wi-Fi, anywhere):

🌐 **Live Global HTTPS URL:**
```
https://easier-riders-waiting-silk.trycloudflare.com
```

- **Android / iOS Installation:** Open the link in Chrome (Android) or Safari (iOS), tap the menu, and choose **"Add to Home screen"** / **"Install App"** to install Bovista as a standalone app with camera support.

---

## ☁️ 2. Free Permanent Cloud Deployment Options

All required deployment configuration files (`Dockerfile`, `Procfile`, `render.yaml`, `docker-compose.yml`, `requirements.txt`) have been generated in the project.

### Option A: Render.com (Recommended - 100% Free)
1. Push this repository to **GitHub**.
2. Go to [dashboard.render.com](https://dashboard.render.com) and click **New +** > **Web Service** (or **Blueprint** using `render.yaml`).
3. Connect your GitHub repository.
4. Render will auto-detect the configuration:
   - **Build Command:** `pip install -r requirements.txt gunicorn`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120`
5. Click **Create Web Service**. Your app will be live permanently at `https://bovista-ai.onrender.com`.

---

### Option B: Hugging Face Spaces (Free 16GB RAM Container)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Choose **Docker** as the Space SDK.
3. Push your repository to the Hugging Face Space git remote.
4. Hugging Face will automatically build and host the `Dockerfile` with zero configuration.

---

### Option C: Railway.app / Koyeb
1. Connect your GitHub repository on [railway.app](https://railway.app) or [koyeb.com](https://koyeb.com).
2. The platform will automatically recognize the `Procfile` / `Dockerfile` and deploy the service.

---

## 🐳 3. Self-Hosted Docker Deployment

To run the container on any Linux server, VPS, or cloud VM:

```bash
# Build and run with Docker Compose
docker-compose up -d --build

# Or run directly with Docker
docker build -t bovista-app .
docker run -d -p 5000:5000 --name bovista-container bovista-app
```

---

## 🔄 4. Launching the Cloudflare Public Tunnel Locally

To start a new global public tunnel at any time on your machine:

```powershell
.\cloudflared.exe tunnel --url http://127.0.0.1:5000
```
