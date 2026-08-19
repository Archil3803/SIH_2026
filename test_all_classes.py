import os
import sys

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
        pred_id = res["predicted_breed"]["id"]
        pred_name = res["predicted_breed"]["name"]
        conf = res["predicted_breed"]["confidence_percent"]
        
        is_match = (pred_id == cls_name)
        if is_match:
            correct += 1
            status = "[MATCH]"
        else:
            status = "[PRED]"

        print(f"{status} True: {cls_name:<25} | Predicted: {pred_name:<25} ({conf}%)")

    print("\n" + "="*60)
    print(f"Sample test complete: {correct}/{total} top-1 matches on sample batch.")
    print("="*60)

if __name__ == "__main__":
    main()
