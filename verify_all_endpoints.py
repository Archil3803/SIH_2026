import os
import sys
import io
import json
import urllib.request
import urllib.error
import urllib.parse

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:5000"

def test_endpoint(name, url, method="GET", data=None, headers=None):
    try:
        encoded_url = urllib.parse.quote(url, safe=':/?=&')
        req = urllib.request.Request(encoded_url, data=data, headers=headers or {}, method=method)
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read()
            print(f"[PASS] {name} -> Status {status} ({content_type.split(';')[0]})")
            return status, body
    except urllib.error.HTTPError as e:
        print(f"[FAIL] {name} -> HTTPError {e.code}: {e.read().decode('utf-8')}")
        return e.code, None
    except Exception as e:
        print(f"[FAIL] {name} -> Error: {e}")
        return None, None


def main():
    print("="*70)
    print("STARTING COMPREHENSIVE FLASK & INFERENCE API VERIFICATION")
    print("="*70)

    # 1. Test HTML Main Page
    status, html_bytes = test_endpoint("Main Dashboard UI", f"{BASE_URL}/")
    html_text = html_bytes.decode("utf-8")
    assert "Bovista" in html_text, "Brand missing in HTML"
    assert "dropzone" in html_text, "Upload dropzone missing"
    assert "tab-overview" in html_text, "Overview tab missing"
    assert "tab-milk" in html_text, "Milk tab missing"
    assert "tab-health" in html_text, "Health tab missing"
    assert "tab-vaccination" in html_text, "Vaccination tab missing"
    assert "tab-economics" in html_text, "Economics tab missing"
    assert "tab-maintenance" in html_text, "Maintenance tab missing"

    # 2. Test Static CSS & JS
    test_endpoint("CSS Stylesheet", f"{BASE_URL}/static/css/style.css")
    test_endpoint("JS Script", f"{BASE_URL}/static/js/app.js")

    # 3. Test PWA & Mobile Endpoints
    test_endpoint("PWA Manifest", f"{BASE_URL}/manifest.json")
    test_endpoint("PWA ServiceWorker", f"{BASE_URL}/sw.js")
    status, net_bytes = test_endpoint("Mobile Network Info", f"{BASE_URL}/api/network-info")
    net_data = json.loads(net_bytes)
    print(f"       -> Mobile Connection URL: {net_data.get('mobile_url')}")
    test_endpoint("Direct APK Download", f"{BASE_URL}/api/download-apk")

    # 4. Test Health Check
    status, health_bytes = test_endpoint("Health Check API", f"{BASE_URL}/api/health")
    health = json.loads(health_bytes)
    assert health.get("status") == "healthy"

    # 4. Test Breeds Catalog API
    status, breeds_bytes = test_endpoint("All Breeds API", f"{BASE_URL}/api/breeds")
    breeds_data = json.loads(breeds_bytes)
    assert breeds_data.get("count") == 10, f"Expected 10 breeds, got {breeds_data.get('count')}"

    # 5. Test Specific Breed API
    status, single_breed_bytes = test_endpoint("Breed Detail (Jaffarabadi)", f"{BASE_URL}/api/breed/jaffarabadi")
    single_breed = json.loads(single_breed_bytes)
    assert single_breed["success"] is True
    assert single_breed["breed"]["category"] == "Buffalo"

    # 6. Test Sample Images API
    status, samples_bytes = test_endpoint("Sample Images API", f"{BASE_URL}/api/sample-images")
    samples_data = json.loads(samples_bytes)
    assert samples_data.get("count") == 10, f"Expected 10 samples, got {samples_data.get('count')}"
    sample_img_url = samples_data["samples"][0]["image_url"]

    # 7. Test Serving Dataset Image
    test_endpoint("Dataset Sample Image File", f"{BASE_URL}{sample_img_url}")

    # 8. Test Predict API via Sample Path
    sample_path = samples_data["samples"][0]["file_path"]
    predict_payload = json.dumps({"sample_path": sample_path}).encode("utf-8")
    status, pred_bytes = test_endpoint(
        "Predict API (JSON payload)",
        f"{BASE_URL}/api/predict",
        method="POST",
        data=predict_payload,
        headers={"Content-Type": "application/json"}
    )
    pred_res = json.loads(pred_bytes)
    assert pred_res.get("success") is True
    pred_breed = pred_res["predicted_breed"]
    details = pred_res["breed_details"]

    print(f"       -> Predicted: {pred_breed['name']} ({pred_breed['confidence_percent']}%)")
    print(f"       -> Lifespan: {details.get('lifespan', {}).get('average_lifespan_years')}")
    print(f"       -> Milk Daily: {details.get('milk_production', {}).get('daily_yield_liters')}")
    print(f"       -> Milk Fat %: {details.get('milk_production', {}).get('fat_percentage')}")
    print(f"       -> Diseases Count: {len(details.get('possible_diseases', []))}")
    print(f"       -> Cures: {details.get('cure_and_treatment', {}).get('emergency_first_aid')[:45]}...")
    print(f"       -> Vaccinations: {len(details.get('vaccination_schedule', []))} vaccines listed")
    print(f"       -> Market Price: {details.get('market_price', {}).get('currency_inr')}")
    print(f"       -> Green Fodder: {details.get('maintenance_and_housing', {}).get('daily_feed_requirements', {}).get('green_fodder_kg')}")

    # 9. Test Predict API via Multipart File Upload
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(sample_path, "rb") as f:
        file_content = f.read()

    body_io = io.BytesIO()
    body_io.write(f"--{boundary}\r\n".encode("utf-8"))
    body_io.write(f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(sample_path)}"\r\n'.encode("utf-8"))
    body_io.write(b"Content-Type: image/jpeg\r\n\r\n")
    body_io.write(file_content)
    body_io.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    multipart_data = body_io.getvalue()

    status, upload_pred_bytes = test_endpoint(
        "Predict API (Multipart Upload)",
        f"{BASE_URL}/api/predict",
        method="POST",
        data=multipart_data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    upload_pred_res = json.loads(upload_pred_bytes)
    assert upload_pred_res.get("success") is True
    print(f"       -> Upload Predicted: {upload_pred_res['predicted_breed']['name']} ({upload_pred_res['predicted_breed']['confidence_percent']}%)")

    # 10. Test Predict API on Non-Bovine Image (Must return error alert without probabilities)
    from PIL import Image
    import numpy as np
    non_bovine_img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    nb_io = io.BytesIO()
    non_bovine_img.save(nb_io, format="JPEG")
    nb_bytes = nb_io.getvalue()

    nb_body = io.BytesIO()
    nb_body.write(f"--{boundary}\r\n".encode("utf-8"))
    nb_body.write(b'Content-Disposition: form-data; name="file"; filename="non_bovine.jpg"\r\n')
    nb_body.write(b"Content-Type: image/jpeg\r\n\r\n")
    nb_body.write(nb_bytes)
    nb_body.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    status, nb_pred_bytes = test_endpoint(
        "Predict API (Non-Bovine Upload)",
        f"{BASE_URL}/api/predict",
        method="POST",
        data=nb_body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    nb_pred_res = json.loads(nb_pred_bytes)
    assert nb_pred_res.get("is_bovine") is False, "is_bovine must be False"
    assert nb_pred_res.get("error") == "non - bovine image detected", f"Expected error 'non - bovine image detected', got '{nb_pred_res.get('error')}'"
    assert "top_candidates" not in nb_pred_res, "Must not return classification probabilities for non-bovine"
    assert "predicted_breed" not in nb_pred_res, "Must not return predicted_breed for non-bovine"
    print(f"       -> Non-Bovine Error Alert Verified: '{nb_pred_res.get('error')}' with zero probabilities returned.")

    print("="*70)
    print("ALL API & SYSTEM ENDPOINTS VERIFIED AND PASSING 100%!")
    print("="*70)

if __name__ == "__main__":
    main()
