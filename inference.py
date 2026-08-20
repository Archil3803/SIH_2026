import os
import io
import json
import base64
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import load_trained_model
from species_detector import SpeciesDetector
from bovine_detector import get_bovine_detector

DEFAULT_MODEL_PATH = "models/cattle_classifier.pth"
DEFAULT_CLASSES_PATH = "models/classes.json"
DEFAULT_BREED_DB_PATH = "data/breed_database.json"

class CattlePredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, classes_path=DEFAULT_CLASSES_PATH, breed_db_path=DEFAULT_BREED_DB_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.classes_path = classes_path
        self.breed_db_path = breed_db_path

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

        # Load Species & Non-Bovine Validator
        self.species_detector = SpeciesDetector(device=self.device)

        # Load Multi-Bovine Object Localization & Instance Detector
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
                "ayrshire_cattle", "brown_swiss_cattle", "chhattisgarhi",
                "holstein_friesian_cattle", "jaffarabadi", "jersey_cattle",
                "marathwada", "red_dane_cattle", "surti", "toda"
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

    def _classify_crop(self, pil_crop, top_k=3):
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

        # Stage 1: Species & Bovine Verification
        species_info = self.species_detector.detect(
            pil_crop,
            fine_grained_conf=max_prob_val,
            fine_grained_entropy=entropy
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
        if max_prob_val < 0.45 or entropy > 1.65:
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

    def predict(self, image_input, top_k=3):
        """
        Multi-bovine detection & per-instance classification pipeline:
        1. Detects all cattle and buffalo instances in the image.
        2. For each detected instance, crops and classifies the exact breed.
        3. Annotates the image with color-coded bounding boxes and breed labels.
        4. If no multi-animal boxes detected, processes the full image.
        """
        pil_img = self.load_pil_image(image_input)
        raw_detections = self.bovine_detector.detect_bovines(pil_img)

        instances = []

        if raw_detections and len(raw_detections) > 0:
            for idx, det in enumerate(raw_detections):
                crop = self.bovine_detector.crop_instance(pil_img, det["box"])
                crop_res = self._classify_crop(crop, top_k=top_k)

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

            # Filter valid instances
            valid_bovines = [inst for inst in instances if inst["is_bovine"]]

            if not valid_bovines:
                # Detections might have been false-positive non-bovines; check whole image
                full_res = self._classify_crop(pil_img, top_k=top_k)
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

        # Case: Zero object detections -> Evaluate whole image
        full_res = self._classify_crop(pil_img, top_k=top_k)

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

        # Single verified bovine (full image)
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


# Global singleton predictor instance for fast caching in web app
_global_predictor = None

def get_predictor():
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = CattlePredictor()
    return _global_predictor
