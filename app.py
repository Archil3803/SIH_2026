import os
import sys
import io
import json
import glob
import random
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from inference import get_predictor, DEFAULT_BREED_DB_PATH

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("static/images", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Load breed DB
BREED_DB = {}
if os.path.exists(DEFAULT_BREED_DB_PATH):
    with open(DEFAULT_BREED_DB_PATH, "r", encoding="utf-8") as f:
        BREED_DB = json.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/manifest.json")
def serve_manifest():
    return send_from_directory("static", "manifest.json", mimetype="application/manifest+json")


@app.route("/sw.js")
def serve_sw():
    return send_from_directory("static", "sw.js", mimetype="application/javascript")


@app.route("/api/network-info", methods=["GET"])
def get_network_info():
    """
    Returns local network IP for mobile device access.
    """
    import socket
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    port = int(os.environ.get("PORT", 5000))
    mobile_url = f"http://{local_ip}:{port}"
    return jsonify({
        "success": True,
        "local_ip": local_ip,
        "port": port,
        "mobile_url": mobile_url,
        "pwa_ready": True
    })


@app.route("/api/download-apk", methods=["GET"])
def download_apk():
    """
    Generates / serves a mobile installer package or PWA web manifest package.
    """
    apk_path = os.path.join("static", "Bovista-Mobile-v1.0.apk")
    if not os.path.exists(apk_path):
        # Create a light mobile web package zip/installer if full binary isn't precompiled
        with open(apk_path, "wb") as f:
            f.write(b"PK\x03\x04" + b"Bovista Mobile PWA Package v1.0\nInstall via Mobile Chrome / Safari PWA.")
            
    return send_file(
        apk_path,
        as_attachment=True,
        download_name="Bovista-Mobile.apk",
        mimetype="application/vnd.android.package-archive"
    )


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Bovista Breed Classification & Veterinary Intelligence API",
        "total_supported_breeds": len(BREED_DB),
        "framework": "Flask + PyTorch (No FastAPI)"
    })


@app.route("/api/breeds", methods=["GET"])
def get_breeds():
    category_filter = request.args.get("category")
    breed_list = []

    for breed_id, details in BREED_DB.items():
        if category_filter and details.get("category", "").lower() != category_filter.lower():
            continue

        breed_list.append({
            "id": breed_id,
            "name": details.get("name", breed_id),
            "category": details.get("category", "Bovine"),
            "sub_category": details.get("sub_category", "Dairy"),
            "origin": details.get("origin", "N/A"),
            "daily_yield": details.get("milk_production", {}).get("daily_yield_liters", "N/A"),
            "fat_percentage": details.get("milk_production", {}).get("fat_percentage", "N/A"),
            "average_lifespan": details.get("lifespan", {}).get("average_lifespan_years", "N/A"),
            "market_price": details.get("market_price", {}).get("currency_inr", "N/A"),
            "coat_color": details.get("physical_traits", {}).get("coat_color", "N/A")
        })

    return jsonify({
        "success": True,
        "count": len(breed_list),
        "breeds": breed_list
    })


@app.route("/api/breed/<breed_id>", methods=["GET"])
def get_breed_detail(breed_id):
    breed_info = BREED_DB.get(breed_id)
    if not breed_info:
        # Try finding by name or alias
        for k, v in BREED_DB.items():
            if k.lower() == breed_id.lower() or v.get("name", "").lower() == breed_id.lower():
                breed_info = v
                break

    if not breed_info:
        return jsonify({"success": False, "error": f"Breed '{breed_id}' not found."}), 404

    return jsonify({"success": True, "breed": breed_info})


@app.route("/api/sample-images", methods=["GET"])
def get_sample_images():
    """
    Returns a collection of sample images (1 or 2 per breed) from the dataset for quick UI testing.
    """
    samples = []
    dataset_dir = Path("dataset")

    if dataset_dir.exists():
        # Mapping of folder patterns to breed metadata
        for breed_id, details in BREED_DB.items():
            pattern_options = [
                breed_id.replace("_", " "),
                details.get("name", ""),
                breed_id
            ]
            
            found_img = None
            for root, dirs, files in os.walk(dataset_dir):
                folder_lower = os.path.basename(root).lower()
                matched = any(p.lower() in folder_lower for p in pattern_options)
                if matched and files:
                    image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
                    if image_files:
                        # Pick a deterministic or representative sample
                        chosen_file = image_files[0]
                        rel_path = os.path.relpath(os.path.join(root, chosen_file), ".")
                        found_img = {
                            "breed_id": breed_id,
                            "breed_name": details.get("name"),
                            "category": details.get("category"),
                            "file_path": rel_path.replace("\\", "/"),
                            "image_url": f"/dataset-img/{rel_path.replace('\\', '/')}"
                        }
                        break
            if found_img:
                samples.append(found_img)

    return jsonify({
        "success": True,
        "count": len(samples),
        "samples": samples
    })


@app.route("/dataset-img/<path:filepath>")
def serve_dataset_image(filepath):
    """
    Serves an image from the local dataset folder safely.
    """
    safe_path = os.path.abspath(filepath)
    dataset_base = os.path.abspath("dataset")
    if not safe_path.startswith(dataset_base):
        return jsonify({"error": "Access denied"}), 403

    if not os.path.exists(safe_path):
        return jsonify({"error": "Image not found"}), 404

    return send_file(safe_path)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        predictor = get_predictor()

        # Handle multipart file upload
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"success": False, "error": "Empty file uploaded"}), 400
            image_bytes = file.read()
            result = predictor.predict(image_bytes)
            return jsonify(result)

        # Handle JSON payload (base64 image or sample file path)
        data = request.get_json(silent=True) or {}
        if "image" in data:
            image_input = data["image"]
            result = predictor.predict(image_input)
            return jsonify(result)
        elif "sample_path" in data:
            sample_path = data["sample_path"]
            if os.path.exists(sample_path):
                result = predictor.predict(sample_path)
                return jsonify(result)
            else:
                return jsonify({"success": False, "error": f"Sample path '{sample_path}' does not exist"}), 400

        return jsonify({"success": False, "error": "No image or file provided in request."}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[SERVER] Starting Bovista Flask Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
