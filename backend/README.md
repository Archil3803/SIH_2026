# Bovista Backend 🧠⚙️

The PyTorch & Flask backend service powering multi-bovine detection, species validation, fine-grained breed classification, and veterinary dossier intelligence.

---

## 📁 Directory Layout

```
backend/
├── app.py                    # Flask REST API server and static frontend bridge
├── inference.py              # PyTorch inference engine & multi-bovine instance analyzer
├── species_detector.py       # Two-tiered species & non-bovine validation
├── bovine_detector.py        # Bovine localization & bounding box segmentation
├── model.py                  # Neural network architecture definitions (MobileNetV3 / ResNet)
├── data_loader.py            # Supervised dataset scanner & PyTorch DataLoader pipeline
├── train.py                  # Model training and fine-tuning workflow
├── predict_cli.py            # Command-line prediction utility
├── requirements.txt          # Python dependencies
├── data/
│   ├── breed_database.json   # Comprehensive 10-breed veterinary knowledge base
│   └── dataset.csv           # Supervised dataset catalog
├── models/
│   ├── cattle_classifier.pth # Trained model weights
│   ├── classes.json          # Standardized class-to-index mappings
│   ├── kfold_evaluation_report.json
│   └── training_history.json
└── tests/
    ├── __init__.py
    ├── test_system.py        # System integrity & non-bovine suppression unit tests
    ├── test_kfold_integrity.py # 5-Fold cross-validation audit
    ├── test_all_classes.py   # Multi-class inference verification
    └── verify_all_endpoints.py # End-to-end API verification suite
```

---

## 🚀 Running the Backend

### Start Server
```bash
python backend/app.py
```
Or from the project root:
```bash
python run.py
```

Server starts by default at `http://127.0.0.1:5000`.

---

## 📡 REST API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the Bovista PWA frontend dashboard |
| `/api/health` | `GET` | Service status, framework info, and supported breed count |
| `/api/breeds` | `GET` | Returns list of all supported breeds with summary stats |
| `/api/breed/<breed_id>` | `GET` | Returns full veterinary & economic dossier for a breed |
| `/api/sample-images` | `GET` | Returns representative sample images for testing |
| `/api/predict` | `POST` | Performs AI inference (multipart upload, base64, or sample path) |
| `/api/network-info` | `GET` | Local network IP & port for mobile device QR pairing |
| `/api/download-apk` | `GET` | Direct download endpoint for Android APK package |
| `/dataset-img/<path>` | `GET` | Safely serves dataset images with path protection |

---

## 🧪 Running Tests

```bash
# Run unit & system tests
python backend/tests/test_system.py

# Run K-Fold integrity evaluation
python backend/tests/test_kfold_integrity.py

# Run live endpoint verification (while server is running)
python backend/tests/verify_all_endpoints.py
```
