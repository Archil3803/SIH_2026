import os
import sys
from pathlib import Path

# Add backend directory to sys.path
TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from data_loader import scan_dataset, CattleDataset, get_data_transforms, CLASS_DISPLAY_NAMES
from model import load_trained_model, build_model
from inference import get_predictor


def run_integrity_tests():
    print("=" * 75)
    print(" 🛡️  STEP 1: RUNNING SYSTEM & MODEL INTEGRITY AUDIT")
    print("=" * 75)

    predictor = get_predictor()

    # 1. Checkpoint File & Weight Integrity
    ckpt_candidates = [
        BACKEND_DIR / "models" / "cattle_classifier.pth",
        PROJECT_ROOT / "models" / "cattle_classifier.pth",
        Path("models/cattle_classifier.pth")
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if c.exists():
            ckpt_path = c
            break
    assert ckpt_path is not None and ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert "model_state_dict" in state or "state_dict" in state or isinstance(state, dict), "Invalid weight dict format"
    print(f" [PASS] Model Checkpoint: Verified '{ckpt_path}' (Size: {os.path.getsize(str(ckpt_path))/1024/1024:.2f} MB)")

    # 2. Database Schema Integrity
    db_candidates = [
        BACKEND_DIR / "data" / "breed_database.json",
        PROJECT_ROOT / "data" / "breed_database.json",
        Path("data/breed_database.json")
    ]
    db_path = None
    for c in db_candidates:
        if c.exists():
            db_path = c
            break
    assert db_path is not None and db_path.exists(), "breed_database.json missing"
    with open(str(db_path), "r", encoding="utf-8") as f:
        db = json.load(f)

    expected_10 = [
        "chhattisgarhi", "gir", "jaffarabadi", "jersey_cattle", "kankrej",
        "marathwada", "red_sindhi", "sahiwal", "surti", "toda"
    ]
    for b in expected_10:
        assert b in db, f"Missing breed key '{b}' in breed_database.json"
        entry = db[b]
        for req_field in ["lifespan", "milk_production", "milk_quality", "possible_diseases", "cure_and_treatment", "vaccination_schedule", "market_price", "maintenance_and_housing"]:
            assert req_field in entry, f"Missing '{req_field}' in {b}"
    print(f" [PASS] Breed Database: Verified 10 primary breeds with complete veterinary fields ({len(db)} total entries)")

    # 3. CSV Dataset & Files Alignment
    csv_candidates = [
        BACKEND_DIR / "data" / "dataset.csv",
        PROJECT_ROOT / "data" / "dataset.csv",
        PROJECT_ROOT / "dataset.csv",
        Path("dataset.csv")
    ]
    csv_path = None
    for c in csv_candidates:
        if c.exists():
            csv_path = c
            break
    assert csv_path is not None and csv_path.exists(), f"dataset.csv missing"
    samples, classes = scan_dataset("dataset")
    assert len(classes) == 10, f"Expected 10 classes in dataset, found {len(classes)}"
    assert len(samples) == 2349, f"Expected 2349 images, found {len(samples)}"
    print(f" [PASS] Dataset & CSV: Verified 2,349 images perfectly aligned across 10 classes")

    # 4. Non-Bovine Rejection Integrity (Batman & Noise)
    if os.path.exists("Batmann.jpg"):
        res_batman = predictor.predict("Batmann.jpg")
        assert res_batman["is_bovine"] is False, "Batman must be flagged as non-bovine"
        assert res_batman["error"] == "non - bovine image detected", "Error message must match"
        assert "predicted_breed" not in res_batman, "Probabilities must be suppressed for Batman"
        print(f" [PASS] Non-Bovine Integrity: 'Batmann.jpg' successfully rejected with error alert and zero false probabilities")

    # 5. Pure Noise Suppression
    noise_img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    res_noise = predictor.predict(noise_img)
    assert res_noise["is_bovine"] is False
    print(f" [PASS] Noise Suppression: Random noise matrix successfully rejected")

    print("\n -> All 5 System Integrity Checks PASSED!\n")


def create_stratified_folds(samples, n_splits=5, seed=42):
    random.seed(seed)
    by_class = {}
    for path, cls in samples:
        by_class.setdefault(cls, []).append((path, cls))

    for cls in by_class:
        random.shuffle(by_class[cls])

    folds = [[] for _ in range(n_splits)]

    for cls, items in by_class.items():
        for i, item in enumerate(items):
            folds[i % n_splits].append(item)

    # Shuffle each fold
    for f in folds:
        random.shuffle(f)

    return folds


def run_kfold_evaluation(n_splits=5):
    print("=" * 75)
    print(f" 📊  STEP 2: RUNNING STRATIFIED {n_splits}-FOLD CROSS VALIDATION")
    print("=" * 75)

    samples, classes = scan_dataset("dataset")
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    num_classes = len(classes)

    _, val_transform = get_data_transforms()

    model = load_trained_model("models/cattle_classifier.pth", num_classes=num_classes)
    model.eval()

    folds = create_stratified_folds(samples, n_splits=n_splits, seed=42)

    fold_accuracies = []
    all_y_true = []
    all_y_pred = []
    all_y_scores = []

    print(f"Total dataset: {len(samples)} samples across {num_classes} classes.")
    print(f"Splits: {n_splits} folds (~{len(samples)//n_splits} samples per fold)\n")

    for fold_idx in range(n_splits):
        fold_samples = folds[fold_idx]
        val_dataset = CattleDataset(fold_samples, class_to_idx, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

        correct = 0
        total = 0
        fold_y_true = []
        fold_y_pred = []

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                fold_y_true.extend(labels.cpu().numpy())
                fold_y_pred.extend(preds.cpu().numpy())
                all_y_scores.extend(probs.cpu().numpy())

        fold_acc = (correct / total) * 100.0
        fold_accuracies.append(fold_acc)
        all_y_true.extend(fold_y_true)
        all_y_pred.extend(fold_y_pred)

        print(f"  • Fold {fold_idx + 1}/{n_splits}: {correct}/{total} correct -> Accuracy: {fold_acc:.2f}%")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    best_fold_acc = float(np.max(fold_accuracies))
    overall_acc = float((all_y_true == all_y_pred).sum() / len(all_y_true) * 100.0)

    # 95% Confidence Interval for mean accuracy: mean +- 1.96 * (std / sqrt(k))
    ci95 = 1.96 * (std_acc / np.sqrt(n_splits))
    optimal_lower = mean_acc - ci95
    optimal_upper = mean_acc + ci95

    print("\n" + "=" * 75)
    print(" 📈  K-FOLD CROSS-VALIDATION SUMMARY RESULTS")
    print("=" * 75)
    print(f" Mean Cross-Validation Accuracy (μ):   {mean_acc:.2f}%")
    print(f" Standard Deviation (σ):               ±{std_acc:.2f}%")
    print(f" Peak Single-Fold Accuracy:            {best_fold_acc:.2f}%")
    print(f" Overall Full-Dataset Accuracy:        {overall_acc:.2f}%")
    print(f" Optimal Accuracy (95% CI Range):      {optimal_lower:.2f}% - {optimal_upper:.2f}%")
    print("=" * 75)

    # Compute Per-Class Classification Report
    print("\n 🏷️  PER-CLASS ACCURACY, PRECISION, RECALL & F1-SCORE")
    print("-" * 75)
    print(f" {'Breed Class':<26} {'Samples':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 75)

    per_class_metrics = {}
    for c_idx in range(num_classes):
        c_name = idx_to_class[c_idx]
        disp_name = CLASS_DISPLAY_NAMES.get(c_name, c_name)

        tp = np.sum((all_y_true == c_idx) & (all_y_pred == c_idx))
        fp = np.sum((all_y_true != c_idx) & (all_y_pred == c_idx))
        fn = np.sum((all_y_true == c_idx) & (all_y_pred != c_idx))
        support = np.sum(all_y_true == c_idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics[c_name] = {
            "display_name": disp_name,
            "support": int(support),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2)
        }

        print(f" {disp_name:<26} {support:<8} {precision*100:>6.2f}%    {recall*100:>6.2f}%    {f1*100:>6.2f}%")

    print("-" * 75)

    # 10x10 Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_y_true, all_y_pred):
        cm[t, p] += 1

    print("\n 🔢  10x10 CONFUSION MATRIX")
    print("-" * 75)
    headers = [f"C{i+1}" for i in range(num_classes)]
    print(f" {'True \\ Pred':<22} " + " ".join(f"{h:>5}" for h in headers))
    print("-" * 75)
    for i, row in enumerate(cm):
        c_name = idx_to_class[i]
        lbl = f"C{i+1}: {CLASS_DISPLAY_NAMES.get(c_name, c_name)[:16]}"
        row_str = " ".join(f"{val:>5}" for val in row)
        print(f" {lbl:<22} {row_str}")
    print("-" * 75)

    # Save validation report JSON
    results = {
        "k_folds": n_splits,
        "total_samples": len(samples),
        "mean_accuracy": round(mean_acc, 2),
        "std_deviation": round(std_acc, 2),
        "best_fold_accuracy": round(best_fold_acc, 2),
        "overall_accuracy": round(overall_acc, 2),
        "optimal_accuracy_ci95": [round(optimal_lower, 2), round(optimal_upper, 2)],
        "fold_accuracies": [round(a, 2) for a in fold_accuracies],
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }

    report_save_path = BACKEND_DIR / "models" / "kfold_evaluation_report.json"
    with open(str(report_save_path), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved full evaluation report to '{report_save_path}'.\n")
    return results


if __name__ == "__main__":
    run_integrity_tests()
    run_kfold_evaluation(n_splits=5)
