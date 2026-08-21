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

# Add backend directory to sys.path so modules like inference, model, species_detector import smoothly
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from inference import get_predictor, DEFAULT_BREED_DB_PATH

# Determine frontend directory
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = BACKEND_DIR / "frontend"

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR),
    static_folder=str(FRONTEND_DIR),
    static_url_path=""
)
CORS(app)

# Ensure folders exist
os.makedirs(str(BACKEND_DIR / "models"), exist_ok=True)
os.makedirs(str(BACKEND_DIR / "data"), exist_ok=True)

# Load breed DB
BREED_DB = {}
breed_db_candidates = [
    DEFAULT_BREED_DB_PATH,
    str(BACKEND_DIR / "data" / "breed_database.json"),
    str(PROJECT_ROOT / "data" / "breed_database.json"),
    str(PROJECT_ROOT / "backend" / "data" / "breed_database.json")
]
for candidate in breed_db_candidates:
    if candidate and os.path.exists(candidate):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                BREED_DB = json.load(f)
                break
        except Exception:
            pass


@app.route("/")
def index():
    """Serves the main application dashboard."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return send_file(str(index_path))
    return jsonify({
        "status": "online",
        "message": "Bovista Backend API is running. Frontend not found at expected path.",
        "api_docs": "/api/health"
    })


@app.route("/manifest.json")
def serve_manifest():
    return send_from_directory(str(FRONTEND_DIR), "manifest.json", mimetype="application/manifest+json")


@app.route("/sw.js")
def serve_sw():
    return send_from_directory(str(FRONTEND_DIR), "sw.js", mimetype="application/javascript")


# Static asset fallback routes for both /static/... and direct /css, /js, /images
@app.route("/static/<path:filename>")
def serve_static_legacy(filename):
    return send_from_directory(str(FRONTEND_DIR), filename)


@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(str(FRONTEND_DIR / "css"), filename)


@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(str(FRONTEND_DIR / "js"), filename)


@app.route("/images/<path:filename>")
def serve_images(filename):
    return send_from_directory(str(FRONTEND_DIR / "images"), filename)


@app.route("/api/network-info", methods=["GET"])
def get_network_info():
    """Returns local network IP for mobile device access."""
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
    """Serves the Android Mobile APK."""
    apk_candidates = [
        FRONTEND_DIR / "BovineAI-Mobile-v1.0.apk",
        FRONTEND_DIR / "static" / "BovineAI-Mobile-v1.0.apk",
        PROJECT_ROOT / "static" / "BovineAI-Mobile-v1.0.apk"
    ]
    for apk_path in apk_candidates:
        if apk_path.exists():
            return send_file(str(apk_path), as_attachment=True, download_name="BovineAI-Mobile-v1.0.apk")
    return jsonify({"success": False, "error": "APK file not found on server"}), 404


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Bovista Breed Classification & Veterinary Intelligence API",
        "total_supported_breeds": len(BREED_DB),
        "framework": "Flask + PyTorch (Decoupled Modular Architecture)"
    })


def get_dataset_base():
    candidates = [
        PROJECT_ROOT / "dataset",
        BACKEND_DIR / "dataset",
        Path("dataset").resolve()
    ]
    for c in candidates:
        if c.exists():
            return c
    return PROJECT_ROOT / "dataset"


def get_breed_media(breed_id, breed_name=""):
    """
    Returns primary image_url, relative file_path, and gallery of image URLs for a given breed.
    """
    dataset_dir = get_dataset_base()
    if not dataset_dir.exists():
        return None, None, []

    folder_map = {
        "chhattisgarhi": "Buffalo/Chhattisgarhi",
        "gir": "Cattle Breeds/Gir",
        "jaffarabadi": "Buffalo/Jaffarabadi",
        "jersey_cattle": "Cattle Breeds/Jersey cattle",
        "kankrej": "Cattle Breeds/Kankrej",
        "marathwada": "Buffalo/marathwada",
        "red_sindhi": "Cattle Breeds/Red_Sindhi",
        "sahiwal": "Cattle Breeds/Sahiwal",
        "surti": "Buffalo/surti",
        "toda": "Buffalo/toda"
    }

    preferred_images = {
        "chhattisgarhi": ["Buffalo_Farming_in_Raipur_Chhattisgarh_.jpg", "Asian_Buffalo_Having_Rest_Stock_Photo_.jpg", "African_buffalo_or_Cape_buffalo_.jpg"],
        "gir": ["Gir_1.JPG", "Gir_10.jPG", "Gir_100.jpeg"],
        "jaffarabadi": ["10_Most_Expensive_Buffalo_Breeds_in_the_.jpg", "32_-_.jpg"],
        "jersey_cattle": ["Jerseycattle0.jpg", "Jerseycattle1.jpg"],
        "kankrej": ["Kankrej_1.JPG", "Kankrej_10.jpg"],
        "marathwada": ["Breeds_of_cattle_buffalo_Agriculture.jpg", "BUFFALO_BREEDS.jpg", "breeds_analyses_breeding_males_.jpg"],
        "red_sindhi": ["Red_Sindhi_1.JPG", "Red_Sindhi_10.png"],
        "sahiwal": ["Sahiwal_1.JPG", "Sahiwal_10.JPG"],
        "surti": ["131_Murrah_Buffalo_for_Dairy_Farming_ (1).jpg", "Best_Murrah_Breed_Buffalo_Price_Near_Me_.jpg"],
        "toda": ["12_Thousand_Buffalo_On_Road_Royalty_.jpg", "141_Buffalo_Types_Stock_Photos_-_Free_.jpg"]
    }

    rel_folder = folder_map.get(breed_id.lower())
    target_dir = (dataset_dir / rel_folder) if rel_folder else None
    
    if not target_dir or not target_dir.exists():
        pattern_options = [breed_id.replace("_", " "), breed_name, breed_id]
        for root, dirs, files in os.walk(dataset_dir):
            folder_lower = os.path.basename(root).lower()
            if any(p and p.lower() in folder_lower for p in pattern_options):
                target_dir = Path(root)
                break

    if target_dir and target_dir.exists():
        files = [f for f in sorted(os.listdir(str(target_dir))) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        if files:
            primary_file = files[0]
            # Check preferred images first
            for pref in preferred_images.get(breed_id.lower(), []):
                if pref in files:
                    primary_file = pref
                    break
            else:
                for f in files:
                    if "_1." in f.lower() or "0." in f.lower():
                        primary_file = f
                        break

            # Calculate relative path from dataset root
            primary_full = target_dir / primary_file
            primary_rel = os.path.relpath(str(primary_full), str(dataset_dir)).replace("\\", "/")
            primary_url = f"/dataset-img/{primary_rel}"

            gallery = []
            for f in files[:8]:
                full_f = target_dir / f
                rel = os.path.relpath(str(full_f), str(dataset_dir)).replace("\\", "/")
                gallery.append({
                    "file_path": rel,
                    "image_url": f"/dataset-img/{rel}"
                })

            return primary_url, primary_rel, gallery

    return None, None, []


@app.route("/api/breeds", methods=["GET"])
def get_breeds():
    category_filter = request.args.get("category")
    breed_list = []

    for breed_id, details in BREED_DB.items():
        if category_filter and details.get("category", "").lower() != category_filter.lower():
            continue

        img_url, file_path, gallery = get_breed_media(breed_id, details.get("name", breed_id))

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
            "coat_color": details.get("physical_traits", {}).get("coat_color", "N/A"),
            "image_url": img_url,
            "file_path": file_path,
            "gallery": gallery
        })

    return jsonify({
        "success": True,
        "count": len(breed_list),
        "breeds": breed_list
    })


@app.route("/api/breed/<breed_id>", methods=["GET"])
def get_breed_detail(breed_id):
    breed_info = BREED_DB.get(breed_id)
    matched_id = breed_id
    if not breed_info:
        # Try finding by name or alias
        for k, v in BREED_DB.items():
            if k.lower() == breed_id.lower() or v.get("name", "").lower() == breed_id.lower():
                breed_info = v
                matched_id = k
                break

    if not breed_info:
        return jsonify({"success": False, "error": f"Breed '{breed_id}' not found."}), 404

    img_url, file_path, gallery = get_breed_media(matched_id, breed_info.get("name", ""))
    enriched_info = dict(breed_info)
    enriched_info["id"] = matched_id
    enriched_info["image_url"] = img_url
    enriched_info["file_path"] = file_path
    enriched_info["gallery"] = gallery

    return jsonify({"success": True, "breed": enriched_info})


@app.route("/api/sample-images", methods=["GET"])
def get_sample_images():
    """
    Returns a collection of sample images (1 or 2 per breed) from the dataset for quick UI testing.
    """
    samples = []
    for breed_id, details in BREED_DB.items():
        img_url, file_path, gallery = get_breed_media(breed_id, details.get("name", ""))
        if img_url:
            samples.append({
                "breed_id": breed_id,
                "breed_name": details.get("name"),
                "category": details.get("category"),
                "file_path": file_path,
                "image_url": img_url
            })

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
    dataset_base = get_dataset_base()
    # Normalize filepath
    clean_subpath = filepath.replace("/", os.sep).replace("\\", os.sep)
    if clean_subpath.startswith("dataset" + os.sep):
        clean_subpath = clean_subpath[len("dataset" + os.sep):]

    target_file = (dataset_base / clean_subpath).resolve()
    base_resolved = dataset_base.resolve()

    try:
        # Prevent directory traversal
        target_file.relative_to(base_resolved)
    except ValueError:
        return jsonify({"error": "Access denied"}), 403

    if not target_file.exists():
        return jsonify({"error": "Image not found"}), 404

    return send_file(str(target_file))


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
            # Resolve sample_path if relative to dataset
            dataset_base = get_dataset_base()
            if os.path.exists(sample_path):
                target_path = sample_path
            elif (dataset_base / sample_path).exists():
                target_path = str(dataset_base / sample_path)
            elif (PROJECT_ROOT / sample_path).exists():
                target_path = str(PROJECT_ROOT / sample_path)
            else:
                target_path = None

            if target_path and os.path.exists(target_path):
                result = predictor.predict(target_path)
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
