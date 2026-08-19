import os
import io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

class SpeciesDetector:
    """
    Vision validator to distinguish cattle/buffalo (bovines)
    from other animals (dogs, cats, horses, sheep, etc.) and non-animal objects.
    """
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load lightweight ImageNet backbone
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.categories = weights.meta["categories"]
        self.model = models.mobilenet_v3_small(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define bovine ImageNet synsets/keywords & agricultural analogs
        self.bovine_keywords = {
            "ox", "water buffalo", "water ox", "asiatic buffalo", "bubalus bubalis",
            "bison", "cow", "bull", "cattle", "bovine", "calves", "calf",
            "zebu", "beefalo", "longhorn", "yak", "oxcart", "plow", "plough", "hartebeest"
        }

        # Animal group taxonomies
        self.canine_keywords = {"dog", "retriever", "terrier", "hound", "shepherd", "bulldog", "poodle", "pug", "husky", "corgi", "chihuahua", "dingo", "wolf", "coyote", "fox"}
        self.feline_keywords = {"cat", "tabby", "persian", "siamese", "egyptian cat", "lion", "tiger", "cheetah", "leopard", "jaguar", "cougar", "lynx", "panther"}
        self.equine_keywords = {"zebra", "donkey", "mule", "stallion", "mare"}
        self.caprine_keywords = {"sheep", "ram", "tup", "goat", "bighorn", "lamb", "ewe"}
        self.avian_keywords = {"bird", "hen", "cock", "rooster", "duck", "goose", "parrot", "eagle", "owl", "penguin", "swan", "ostrich", "peacock"}

    def _classify_category(self, label_lower):
        for kw in self.bovine_keywords:
            if kw in label_lower:
                return "Bovine (Cattle / Buffalo)", "bovine"
        for kw in self.canine_keywords:
            if kw in label_lower:
                return "Canine", "canine"
        for kw in self.feline_keywords:
            if kw in label_lower:
                return "Feline", "feline"
        for kw in self.equine_keywords:
            if kw in label_lower:
                return "Equine", "equine"
        for kw in self.caprine_keywords:
            if kw in label_lower:
                return "Caprine", "caprine"
        for kw in self.avian_keywords:
            if kw in label_lower:
                return "Avian", "avian"
        
        return "Non-Bovine", "other"

    def detect(self, pil_image, fine_grained_conf=None, fine_grained_entropy=None):
        """
        Evaluates whether the input image is a Bovine (Cattle/Buffalo) or a non-bovine image.
        """
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(probs, 5)
        
        bovine_prob_sum = 0.0
        max_non_bovine_prob = 0.0
        strong_non_bovine = False

        for prob, idx in zip(top_probs, top_indices):
            raw_label = self.categories[idx.item()].lower()
            _, animal_type = self._classify_category(raw_label)
            p_val = prob.item()

            if animal_type == "bovine":
                bovine_prob_sum += p_val
            else:
                if p_val > max_non_bovine_prob:
                    max_non_bovine_prob = p_val
                
                # Check for strongly unambiguous non-bovines (cars, electronics, household, carnivores, birds, etc.)
                if any(k in raw_label for k in ["dog", "retriever", "terrier", "cat", "car", "plane", "truck", "bird", "laptop", "chair", "bottle", "pizza", "building", "guitar", "phone"]):
                    if p_val > 0.40:
                        strong_non_bovine = True

        top_raw_label = self.categories[top_indices[0].item()].lower()
        is_top_bovine = any(kw in top_raw_label for kw in self.bovine_keywords)

        # 1. Definite bovine detection in ImageNet
        if is_top_bovine or bovine_prob_sum > 0.20:
            return {
                "is_bovine": True,
                "bovine_confidence": round(max(top_probs[0].item(), bovine_prob_sum) * 100, 2),
                "detected_subject": "Bovine (Cattle / Buffalo)",
                "detected_type": "bovine"
            }

        # 2. Strong non-bovine detected with minimal bovine probability
        if strong_non_bovine and bovine_prob_sum < 0.05 and (fine_grained_conf is None or fine_grained_conf < 0.50):
            return {
                "is_bovine": False,
                "error": "non - bovine image detected",
                "message": "non - bovine image detected"
            }

        # 3. If fine-grained model confirms bovine dataset match (e.g. dairy cattle with spotted/brown coat)
        if fine_grained_conf is not None and fine_grained_conf >= 0.35:
            return {
                "is_bovine": True,
                "bovine_confidence": round(fine_grained_conf * 100, 2),
                "detected_subject": "Bovine (Cattle / Buffalo)",
                "detected_type": "bovine"
            }

        # 4. Out of distribution / noise / solid / other non-bovine
        return {
            "is_bovine": False,
            "error": "non - bovine image detected",
            "message": "non - bovine image detected"
        }
