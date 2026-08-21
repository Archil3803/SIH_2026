import os
import io
import json
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import load_trained_model
from species_detector import SpeciesDetector
from bovine_detector import get_bovine_detector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

def _resolve_path(rel_path):
    candidates = [
        os.path.join(BASE_DIR, rel_path),
        os.path.join(PROJECT_ROOT, rel_path),
        os.path.abspath(rel_path)
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, rel_path)

DEFAULT_MODEL_PATH = _resolve_path("models/cattle_classifier.pth")
DEFAULT_CLASSES_PATH = _resolve_path("models/classes.json")
DEFAULT_BREED_DB_PATH = _resolve_path("data/breed_database.json")

class CattlePredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, classes_path=DEFAULT_CLASSES_PATH, breed_db_path=DEFAULT_BREED_DB_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = _resolve_path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.classes_path = _resolve_path(classes_path) if classes_path else DEFAULT_CLASSES_PATH
        self.breed_db_path = _resolve_path(breed_db_path) if breed_db_path else DEFAULT_BREED_DB_PATH

        # Load class mappings
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.display_names = {}
        self._load_classes()

        # Load Breed Knowledge Base
        self.breed_db = {}
        self._load_breed_database()

        # Load PyTorch Model for fine-grained breed classification
        num_classes = len(self.class_to_idx) if self.class_to_idx else 10
        self.model = load_trained_model(
            model_path=self.model_path,
            num_classes=num_classes,
            device=self.device
        )

        # Load Species, Human & Cartoon Non-Bovine Validator
        self.species_detector = SpeciesDetector(device=self.device)

        # Load Multi-Bovine & Entity Object Localization Detector
        self.bovine_detector = get_bovine_detector()

        # Transforms for inference
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_classes(self):
        if os.path.exists(self.classes_path):
            with open(self.classes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.class_to_idx = data.get("class_to_idx", {})
                self.idx_to_class = {int(k) if k.isdigit() else k: v for k, v in data.get("idx_to_class", {}).items()}
                self.display_names = data.get("display_names", {})
        else:
            fallback_classes = [
                "chhattisgarhi", "gir", "jaffarabadi", "jersey_cattle",
                "kankrej", "marathwada", "red_sindhi", "sahiwal",
                "surti", "toda"
            ]
            self.class_to_idx = {cls: idx for idx, cls in enumerate(fallback_classes)}
            self.idx_to_class = {idx: cls for idx, cls in enumerate(fallback_classes)}

    def _load_breed_database(self):
        if os.path.exists(self.breed_db_path):
            with open(self.breed_db_path, "r", encoding="utf-8") as f:
                self.breed_db = json.load(f)
        else:
            print(f"Warning: Breed DB '{self.breed_db_path}' not found.")

    def load_pil_image(self, image_input):
        """
        Loads and returns a PIL RGB Image from file path, bytes, or base64.
        """
        if isinstance(image_input, str):
            if image_input.startswith("data:image"):
                base64_data = image_input.split(",")[1]
                image_bytes = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            elif os.path.exists(image_input):
                return Image.open(image_input).convert("RGB")
            else:
                raise ValueError(f"Invalid image path: {image_input}")
        elif isinstance(image_input, (bytes, bytearray)):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError("Unsupported image input type.")

    def _pil_to_base64(self, pil_img, format="JPEG", quality=90):
        """
        Converts a PIL image into a base64 Data URL.
        """
        buffered = io.BytesIO()
        pil_img.save(buffered, format=format, quality=quality)
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{b64_str}"

    def _classify_crop(self, pil_crop, top_k=3, is_detected_cow=False, is_detected_person=False):
        """
        Runs fine-grained species verification and 10-breed classification on an individual crop.
        """
        tensor = self.transform(pil_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)[0]
            max_prob, _ = torch.max(probabilities, dim=0)
            max_prob_val = max_prob.item()
            entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum().item()

        # Stage 1: Species, Human & Cartoon Verification
        species_info = self.species_detector.detect(
            pil_crop,
            fine_grained_conf=max_prob_val,
            fine_grained_entropy=entropy,
            is_detected_cow=is_detected_cow,
            is_detected_person=is_detected_person
        )

        if not species_info.get("is_bovine", False):
            return {
                "success": False,
                "is_bovine": False,
                "is_known_breed": False,
                "error": "non - bovine image detected",
                "message": "non - bovine image detected"
            }

        # Stage 2: In-Dataset Breed Verification
        if max_prob_val < 0.30 or entropy > 2.05:
            return {
                "success": False,
                "is_bovine": True,
                "is_known_breed": False,
                "error": "the given breed does not exists in our data",
                "message": "the given breed does not exists in our data"
            }

        # Stage 3: Fine-grained Bovine Breed Classification
        top_k = min(top_k, len(probabilities))
        top_probs, top_indices = torch.topk(probabilities, top_k)

        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            idx_int = idx.item()
            class_key = self.idx_to_class.get(idx_int, str(idx_int))
            display_name = self.display_names.get(class_key, class_key.replace("_", " ").title())
            confidence_pct = round(prob.item() * 100.0, 2)

            top_predictions.append({
                "class_id": class_key,
                "display_name": display_name,
                "confidence_score": round(prob.item(), 4),
                "confidence_percent": confidence_pct
            })

        best_prediction = top_predictions[0]
        best_class_id = best_prediction["class_id"]

        breed_info = self.breed_db.get(best_class_id, {
            "id": best_class_id,
            "name": best_prediction["display_name"],
            "description": "Breed profile available in standard catalog."
        })

        return {
            "success": True,
            "is_bovine": True,
            "is_known_breed": True,
            "predicted_breed": {
                "id": best_class_id,
                "name": breed_info.get("name", best_prediction["display_name"]),
                "category": breed_info.get("category", "Bovine"),
                "sub_category": breed_info.get("sub_category", "Dairy"),
                "confidence_score": best_prediction["confidence_score"],
                "confidence_percent": best_prediction["confidence_percent"],
                "origin": breed_info.get("origin", "N/A"),
                "native_region": breed_info.get("native_region", "N/A")
            },
            "top_candidates": top_predictions,
            "breed_details": breed_info,
            "species_verification": {
                "is_bovine": True,
                "detected_subject": "Bovine (Cattle / Buffalo)"
            }
        }

    def compute_blur_score(self, pil_img):
        """
        Computes image sharpness metric using Laplacian variance.
        Clear dataset images score > 200.0, while blurry images score < 60.0.
        """
        try:
            gray = pil_img.convert("L")
            gray.thumbnail((600, 600), Image.Resampling.BILINEAR)
            arr = np.array(gray, dtype=np.float64)
            if arr.shape[0] < 3 or arr.shape[1] < 3:
                return 0.0
            lap = (
                arr[2:, 1:-1] + arr[:-2, 1:-1] +
                arr[1:-1, 2:] + arr[1:-1, :-2] -
                4.0 * arr[1:-1, 1:-1]
            )
            return float(np.var(lap))
        except Exception:
            return 1000.0

    def predict(self, image_input, top_k=3, blur_threshold=60.0):
        """
        Multi-layer vision pipeline:
        1. Checks for image blurriness (prompts reupload if blurry).
        2. Detects and rejects cartoon/illustrated/synthetic images.
        3. Detects cattle/buffaloes and humans in scene.
        4. Isolates and classifies individual bovine instances.
        5. Rejects human portraits, other animals, and non-bovines.
        """
        pil_img = self.load_pil_image(image_input)

        # 1. Blur Detection
        blur_score = self.compute_blur_score(pil_img)
        if blur_score < blur_threshold:
            return {
                "success": False,
                "is_blurry": True,
                "blur_score": round(blur_score, 2),
                "is_bovine": False,
                "is_known_breed": False,
                "total_detected": 0,
                "instances": [],
                "error": "reupload a clear image",
                "error_code": "BLURRY_IMAGE",
                "alert_message": "reupload a clear image",
                "message": "reupload a clear image"
            }

        # 2. Entity Detection (Bovines & Humans)
        entities = self.bovine_detector.detect_entities(pil_img)
        raw_bovines = entities.get("bovines", [])
        raw_humans = entities.get("humans", [])

        # 3. Cartoon / Synthetic Image Filter (checked unless clear live animal detected)
        cartoon_metrics = self.species_detector.analyze_cartoon_features(pil_img)
        if cartoon_metrics["is_cartoon"] and not raw_bovines:
            return {
                "success": False,
                "is_bovine": False,
                "is_known_breed": False,
                "total_detected": 0,
                "instances": [],
                "error": "non - bovine image detected",
                "error_code": "NON_BOVINE_IMAGE",
                "alert_message": "non - bovine image detected",
                "message": "non - bovine image detected"
            }

        # 4. Human-only image rejection
        if raw_humans and not raw_bovines:
            return {
                "success": False,
                "is_bovine": False,
                "is_known_breed": False,
                "total_detected": 0,
                "instances": [],
                "error": "non - bovine image detected",
                "error_code": "NON_BOVINE_IMAGE",
                "alert_message": "non - bovine image detected",
                "message": "non - bovine image detected"
            }

        # 5. Multi-Bovine Detections Present
        if raw_bovines and len(raw_bovines) > 0:
            instances = []
            for idx, det in enumerate(raw_bovines):
                crop = self.bovine_detector.crop_instance(pil_img, det["box"])
                crop_res = self._classify_crop(crop, top_k=top_k, is_detected_cow=True)

                breed_name = crop_res.get("predicted_breed", {}).get("name") if crop_res.get("success") else "Bovine (Out-of-dataset)" if crop_res.get("is_bovine") else "Non-Bovine"
                conf_pct = crop_res.get("predicted_breed", {}).get("confidence_percent") if crop_res.get("success") else None

                inst_obj = {
                    "instance_id": idx + 1,
                    "box": det["box"],
                    "box_normalized": det["box_normalized"],
                    "detector_score": det["score"],
                    "detector_label": det["label"],
                    "breed_name": breed_name,
                    "confidence_percent": conf_pct,
                    "crop_image": self._pil_to_base64(crop),
                    "is_bovine": crop_res.get("is_bovine", False),
                    "is_known_breed": crop_res.get("is_known_breed", False),
                    "success": crop_res.get("success", False),
                    "error": crop_res.get("error"),
                    "message": crop_res.get("message"),
                    "predicted_breed": crop_res.get("predicted_breed"),
                    "top_candidates": crop_res.get("top_candidates", []),
                    "breed_details": crop_res.get("breed_details", {}),
                    "species_verification": crop_res.get("species_verification", {"is_bovine": crop_res.get("is_bovine", False)})
                }
                instances.append(inst_obj)

            valid_bovines = [inst for inst in instances if inst["is_bovine"]]

            if not valid_bovines:
                # Detections might have been false positive non-bovines; check full image
                full_res = self._classify_crop(pil_img, top_k=top_k, is_detected_cow=False, is_detected_person=bool(raw_humans))
                if not full_res.get("is_bovine", False):
                    return {
                        "success": False,
                        "is_bovine": False,
                        "is_known_breed": False,
                        "total_detected": 0,
                        "instances": [],
                        "error": "non - bovine image detected",
                        "message": "non - bovine image detected"
                    }

            # Generate annotated image with bounding boxes
            annotated_pil = self.bovine_detector.draw_annotated_image(pil_img, instances)
            annotated_b64 = self._pil_to_base64(annotated_pil)

            # Select primary animal (first known breed or highest confidence)
            known_instances = [inst for inst in instances if inst.get("is_known_breed") and inst.get("success")]
            primary = known_instances[0] if known_instances else instances[0]

            return {
                "success": any(inst.get("success", False) for inst in instances),
                "is_bovine": True,
                "is_known_breed": any(inst.get("is_known_breed", False) for inst in instances),
                "total_detected": len(instances),
                "instances": instances,
                "annotated_image": annotated_b64,
                "predicted_breed": primary.get("predicted_breed"),
                "top_candidates": primary.get("top_candidates", []),
                "breed_details": primary.get("breed_details", {}),
                "species_verification": {
                    "is_bovine": True,
                    "detected_subject": f"{len(instances)} Bovine(s) Detected"
                }
            }

        # 6. Zero object detections -> Evaluate whole image
        full_res = self._classify_crop(pil_img, top_k=top_k, is_detected_cow=False, is_detected_person=bool(raw_humans))

        if not full_res.get("is_bovine", False):
            return {
                "success": False,
                "is_bovine": False,
                "is_known_breed": False,
                "total_detected": 0,
                "instances": [],
                "error": "non - bovine image detected",
                "message": "non - bovine image detected"
            }

        if not full_res.get("is_known_breed", False) or not full_res.get("success", False):
            return {
                "success": False,
                "is_bovine": True,
                "is_known_breed": False,
                "total_detected": 1,
                "instances": [{
                    "instance_id": 1,
                    "box": [0, 0, pil_img.width, pil_img.height],
                    "box_normalized": [0.0, 0.0, 1.0, 1.0],
                    "is_bovine": True,
                    "is_known_breed": False,
                    "success": False,
                    "error": "the given breed does not exists in our data",
                    "message": "the given breed does not exists in our data"
                }],
                "error": "the given breed does not exists in our data",
                "message": "the given breed does not exists in our data"
            }

        # Single verified photographic bovine (full image)
        w, h = pil_img.size
        single_inst = {
            "instance_id": 1,
            "box": [0, 0, w, h],
            "box_normalized": [0.0, 0.0, 1.0, 1.0],
            "detector_score": 1.0,
            "detector_label": "cow",
            "breed_name": full_res["predicted_breed"]["name"],
            "confidence_percent": full_res["predicted_breed"]["confidence_percent"],
            "crop_image": self._pil_to_base64(pil_img),
            "is_bovine": True,
            "is_known_breed": True,
            "success": True,
            "predicted_breed": full_res["predicted_breed"],
            "top_candidates": full_res["top_candidates"],
            "breed_details": full_res["breed_details"],
            "species_verification": full_res["species_verification"]
        }

        annotated_pil = self.bovine_detector.draw_annotated_image(pil_img, [single_inst])
        annotated_b64 = self._pil_to_base64(annotated_pil)

        return {
            "success": True,
            "is_bovine": True,
            "is_known_breed": True,
            "total_detected": 1,
            "instances": [single_inst],
            "annotated_image": annotated_b64,
            "predicted_breed": full_res["predicted_breed"],
            "top_candidates": full_res["top_candidates"],
            "breed_details": full_res["breed_details"],
            "species_verification": full_res["species_verification"]
        }


# Global singleton predictor instance
_global_predictor = None

def get_predictor():
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = CattlePredictor()
    return _global_predictor
