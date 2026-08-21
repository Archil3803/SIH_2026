"""
Bovista Root Application Bridge (Backward Compatibility Wrapper)
Exposes `app` from `backend.app` for WSGI servers (Gunicorn) and direct execution (`python app.py`).
"""
import os
import sys
from pathlib import Path

# Ensure backend and root paths are available
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[SERVER] Starting Bovista Flask Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
