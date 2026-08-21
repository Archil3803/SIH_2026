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

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from inference import CattlePredictor
from data_loader import scan_dataset

def main():
    print("Testing end-to-end inference across dataset...")
    predictor = CattlePredictor()
    samples, classes = scan_dataset("dataset")

    # Pick 1 sample from each class
    by_class = {}
    for path, cls in samples:
        if cls not in by_class:
            by_class[cls] = path

    print(f"Loaded {len(by_class)} representative test images across {len(classes)} classes.\n")

    correct = 0
    total = len(by_class)

    for cls_name, img_path in by_class.items():
        res = predictor.predict(img_path, top_k=3)
        pred_breed = res.get("predicted_breed")
        if pred_breed:
            pred_id = pred_breed.get("id")
            pred_name = pred_breed.get("name")
            conf = pred_breed.get("confidence_percent")
            is_match = (pred_id == cls_name)
            if is_match:
                correct += 1
                status = "[MATCH]"
            else:
                status = "[PRED]"
            print(f"{status} True: {cls_name:<25} | Predicted: {pred_name:<25} ({conf}%)")
        else:
            status = "[WARN]"
            err = res.get("error", "No prediction")
            print(f"{status} True: {cls_name:<25} | Output: {err}")

    print("\n" + "="*60)
    print(f"Sample test complete: {correct}/{total} top-1 matches on sample batch.")
    print("="*60)

if __name__ == "__main__":
    main()
