# Bovista Frontend 🎨✨

The modern, glassmorphic Progressive Web App (PWA) client for Bovista — AI-Powered Veterinary Intelligence and Bovine Breed Classifier.

---

## 📁 Directory Layout

```
frontend/
├── index.html                # Single-page web application dashboard & PWA container
├── css/
│   └── style.css             # Glassmorphism design system, responsive breakpoints, animations
├── js/
│   ├── app.js                # Frontend state management, camera capture, API client
│   └── qrcode.min.js         # Mobile browser QR code generator
├── images/
│   ├── Bovista.jpeg          # Brand logo & mascot asset
│   ├── favicon.ico           # Web favicon
│   ├── icon-192.png          # PWA 192x192 icon
│   ├── icon-512.png          # PWA 512x512 icon
│   └── breeds/               # Breed illustration assets
├── manifest.json             # PWA web manifest specification
├── sw.js                     # Offline cache & Service Worker
└── BovineAI-Mobile-v1.0.apk  # Android APK distribution package
```

---

## 🚀 Running the Frontend

### Option 1: Integrated with Flask Backend (Recommended)
When you start the backend server, the Flask server automatically serves this frontend at:
```
http://127.0.0.1:5000/
```

### Option 2: Standalone Static Server (Decoupled Dev Mode)
You can run this frontend folder using any static web server:
```bash
# Using Python http.server
cd frontend
python -m http.server 3000

# Using Node.js live-server or npx serve
npx serve .
```
Open `http://localhost:3000`. The frontend will automatically detect that it's running on a separate port and communicate with the backend at `http://127.0.0.1:5000/api` via CORS!

---

## ⚙️ Configuration

If your backend is hosted on a remote server or custom URL, you can configure it globally in `window.BOVISTA_API_URL`:
```html
<script>
  window.BOVISTA_API_URL = "https://your-backend-api.onrender.com";
</script>
```
