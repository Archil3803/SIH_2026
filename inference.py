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

    def predict(self, image_input, top_k=3):
        """
        Multi-tier verification & prediction pipeline:
        1. Validates whether the image is a bovine (cattle or buffalo) vs non-bovine.
           - If non-bovine: error 'non - bovine image detected' (no classification probabilities).
        2. If bovine, checks if the breed is present in our 10-breed dataset.
           - If out-of-dataset breed: error 'the given breed does not exists in our data'.
        3. If present in dataset: performs fine-grained classification and returns full veterinary dossier.
        """
        pil_img = self.load_pil_image(image_input)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Compute fine-grained breed probabilities & entropy
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)[0]
            max_prob, _ = torch.max(probabilities, dim=0)
            max_prob_val = max_prob.item()
            entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum().item()

        # Stage 1: Species & Bovine Verification
        species_info = self.species_detector.detect(
            pil_img,
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
        # If the model cannot match the image with high confidence to our 10 dataset breeds
        if max_prob_val < 0.45 or entropy > 1.65:
            return {
                "success": False,
                "is_bovine": True,
                "is_known_breed": False,
                "error": "the given breed does not exists in our data",
                "message": "the given breed does not exists in our data"
            }

        # Stage 3: Fine-grained Bovine Breed Classification for Verified Dataset Breeds
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

        # Fetch detailed veterinary knowledge dossier
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


# Global singleton predictor instance for fast caching in web app
_global_predictor = None

def get_predictor():
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = CattlePredictor()
    return _global_predictor
