"""
Bovista Application Entrypoint
Launches the Flask backend server which automatically serves the decoupled frontend.
"""
import os
import sys
from pathlib import Path

# Ensure root and backend directories are in sys.path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"============================================================")
    print(f" 🐄 Bovista - Veterinary Intelligence & Breed Classifier")
    print(f" 🌐 Running on: http://127.0.0.1:{port}")
    print(f" 📁 Frontend:   {ROOT_DIR / 'frontend'}")
    print(f" 📁 Backend:    {BACKEND_DIR}")
    print(f"============================================================")
    app.run(host="0.0.0.0", port=port, debug=False)
