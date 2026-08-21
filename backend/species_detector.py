import os
import io
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
import numpy as np
from torchvision import models, transforms


class SpeciesDetector:
    """
    Multi-layer vision validator to distinguish genuine photographic cattle/buffalo (bovines)
    from cartoonized/synthetic images, humans (faces/portraits/people),
    non-bovine animals (dogs, cats, horses, sheep, etc.), and inanimate objects.
    """
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load lightweight ImageNet-1k backbone
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

        # 1. Exact ImageNet synset categories for Bovines & agricultural livestock
        self.exact_bovine_classes = {
            "ox", "water buffalo", "bison", "sorrel", "hartebeest", "oxcart", "plow"
        }
        self.adjacent_livestock_classes = {
            "tusker", "indian elephant", "african elephant", "hog", "warthog", "ram", "bighorn", "ibex", "hippopotamus", "arabian camel"
        }

        # 2. Human attire, persona, facial accessories & gear categories
        self.human_classes = {
            "academic gown", "apron", "backpack", "band aid", "barbell", "bath towel",
            "bearskin", "bikini", "bonnet", "bow tie", "brassiere", "breastplate", "cardigan",
            "cloak", "clog", "cowboy boot", "cowboy hat", "crutch", "diaper", "dumbbell",
            "eyeglass", "face powder", "feather boa", "fur coat", "gasmask", "gown", "groom",
            "hair spray", "hair slide", "holster", "jean", "jersey", "kimono", "knee pad",
            "lab coat", "lipstick", "loafer", "maillot", "mask", "military uniform", "miniskirt",
            "mitten", "monocle", "mortarboard", "neck brace", "necklace", "necktie", "oxygen mask",
            "pajama", "perfume", "poncho", "purse", "running shoe", "sandal", "sarong",
            "scuba diver", "seat belt", "shield", "shoe", "shower cap", "ski mask", "sock",
            "sombrero", "stage", "stethoscope", "stole", "suit", "sunglass", "sunglasses",
            "sunscreen", "sweatshirt", "swimming cap", "swimming trunks", "treadmill", "trench coat",
            "turban", "vestment", "wallet", "wig", "yarmulke"
        }

        # 3. Cartoon, comic, artwork, document, digital screen, toy & synthetic object categories
        self.cartoon_art_classes = {
            "comic book", "book jacket", "jigsaw puzzle", "crossword puzzle", "envelope",
            "menu", "web site", "poster", "paper towel", "toilet tissue", "paintbrush",
            "crayon", "pencil box", "pencil sharpener", "rubber eraser", "binder",
            "carton", "packet", "toyshop", "puppet", "marionette", "balloon", "teddy bear",
            "pinwheel", "kite", "origami", "traffic light", "street sign", "cash machine",
            "vending machine", "screen", "monitor", "television", "cellular telephone",
            "ipod", "cd player", "loudspeaker", "cassette", "cassette player", "modem"
        }

        # 4. Animal group taxonomies (Exact keyword matching)
        self.canine_keywords = {"dog", "retriever", "terrier", "hound", "shepherd", "bulldog", "poodle", "pug", "husky", "corgi", "chihuahua", "dingo", "wolf", "coyote", "fox", "boxer", "bull mastiff", "foxhound"}
        self.feline_keywords = {"cat", "tabby", "persian", "siamese", "egyptian cat", "lion", "tiger", "cheetah", "leopard", "jaguar", "cougar", "lynx", "panther"}
        self.equine_keywords = {"zebra", "donkey", "mule", "stallion", "mare"}
        self.caprine_keywords = {"sheep", "tup", "goat", "lamb", "ewe"}
        self.avian_keywords = {"bird", "hen", "cock", "rooster", "duck", "goose", "parrot", "eagle", "owl", "penguin", "swan", "ostrich", "peacock", "flamingo", "vulture"}

    def analyze_cartoon_features(self, pil_image):
        """
        Extracts visual, texture, and color metrics to identify cartoonized,
        cel-shaded, anime, line-art, or digital illustration images.
        """
        img_rgb = pil_image.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32)

        # 1. Color Quantization Richness (16-level RGB -> 4096 color space)
        quant = (arr // 16).astype(np.int32)
        flat_colors = quant[:, :, 0] * 256 + quant[:, :, 1] * 16 + quant[:, :, 2]
        unique_colors = len(np.unique(flat_colors))

        # 2. Local texture variance & Cel-Shading flat region ratio
        gray = np.array(img_rgb.convert("L"), dtype=np.float32)
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)

        blurred = np.array(img_rgb.convert("L").filter(ImageFilter.BoxBlur(2)), dtype=np.float32)
        local_diff = np.abs(gray - blurred)

        flat_pixels = (local_diff < 1.8) & (grad_mag < 3.0)
        flat_ratio = float(np.mean(flat_pixels))

        dark_strokes = (gray < 45.0) & (grad_mag > 30.0)
        dark_stroke_ratio = float(np.mean(dark_strokes))

        is_cartoon = False
        reasons = []

        if unique_colors < 20:
            is_cartoon = True
            reasons.append(f"Severely restricted color palette ({unique_colors} quantized colors)")
        elif flat_ratio > 0.70 and unique_colors < 150:
            is_cartoon = True
            reasons.append(f"Dominant cel-shaded flat regions (flat_ratio={flat_ratio:.2f})")
        elif flat_ratio > 0.40 and dark_stroke_ratio > 0.012 and unique_colors < 120:
            is_cartoon = True
            reasons.append(f"Line-art contour strokes enclosing flat patches (flat={flat_ratio:.2f}, stroke={dark_stroke_ratio:.3f})")

        return {
            "is_cartoon": is_cartoon,
            "unique_colors": unique_colors,
            "flat_ratio": round(flat_ratio, 3),
            "dark_stroke_ratio": round(dark_stroke_ratio, 4),
            "reasons": reasons
        }

    def analyze_human_features(self, pil_image):
        """
        Computes skin chromaticity and human biometric distribution in YCbCr/HSV space.
        """
        img_rgb = pil_image.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128.0
        cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128.0

        skin_mask = (cb >= 77.0) & (cb <= 127.0) & (cr >= 133.0) & (cr <= 173.0) & (r > g) & (g > b) & ((r - g) >= 15.0)
        skin_ratio = float(np.mean(skin_mask))

        return {
            "skin_ratio": round(skin_ratio, 3),
            "has_prominent_skin": skin_ratio > 0.22
        }

    def detect(self, pil_image, fine_grained_conf=None, fine_grained_entropy=None, is_detected_cow=False, is_detected_person=False):
        """
        Evaluates whether the input image is a genuine photographic Bovine (Cattle/Buffalo),
        or a non-bovine image (human, cartoon, non-bovine animal, or inanimate object).
        """
        if hasattr(pil_image, "convert"):
            pil_image = pil_image.convert("RGB")

        # 1. Deep Semantic Inference with ImageNet Backbone
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(probs, 5)

        bovine_prob_sum = 0.0
        adjacent_livestock_prob_sum = 0.0
        human_prob_sum = 0.0
        cartoon_art_prob_sum = 0.0
        non_bovine_animal_prob_sum = 0.0

        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            raw_label = self.categories[idx.item()].lower()
            p_val = prob.item()
            top_predictions.append((raw_label, p_val))

            if raw_label in self.exact_bovine_classes:
                bovine_prob_sum += p_val
            elif raw_label in self.adjacent_livestock_classes:
                adjacent_livestock_prob_sum += p_val
            elif raw_label in self.human_classes:
                human_prob_sum += p_val
            elif raw_label in self.cartoon_art_classes:
                cartoon_art_prob_sum += p_val
            elif any(k in raw_label for k in self.canine_keywords | self.feline_keywords | self.equine_keywords | self.caprine_keywords | self.avian_keywords):
                non_bovine_animal_prob_sum += p_val

        top_label = top_predictions[0][0]
        top_prob = top_predictions[0][1]
        animal_prob = probs[:398].sum().item()

        # 2. Check for Direct Human Detection from Object Detector or Human Class Dominance
        human_bio = self.analyze_human_features(pil_image)
        is_human_scene = (
            (is_detected_person and not is_detected_cow) or
            top_label in self.human_classes or
            (human_prob_sum > 0.15 and bovine_prob_sum < 0.05) or
            (human_bio["has_prominent_skin"] and bovine_prob_sum < 0.05 and not is_detected_cow)
        )

        if is_human_scene:
            return {
                "is_bovine": False,
                "is_cartoon": False,
                "is_human": True,
                "detected_subject": "Human / Person",
                "error": "non - bovine image detected",
                "message": "non - bovine image detected"
            }

        # 3. Check for Cartoon / Synthetic / Illustrated images
        cartoon_metrics = self.analyze_cartoon_features(pil_image)
        is_cartoon_artwork = (
            (cartoon_metrics["is_cartoon"] and bovine_prob_sum < 0.10 and not is_detected_cow) or
            top_label in self.cartoon_art_classes or
            (cartoon_art_prob_sum > 0.20 and bovine_prob_sum < 0.05)
        )

        if is_cartoon_artwork:
            return {
                "is_bovine": False,
                "is_cartoon": True,
                "is_human": False,
                "detected_subject": "Cartoon / Synthetic Illustration",
                "error": "non - bovine image detected",
                "message": "non - bovine image detected"
            }

        # 4. Check for Inanimate / Manufactured Objects in ImageNet (classes 398..999)
        if animal_prob < 0.08 and bovine_prob_sum < 0.02:
            return {
                "is_bovine": False,
                "is_cartoon": False,
                "is_human": False,
                "detected_subject": "Inanimate / Non-Animal Object",
                "error": "non - bovine image detected",
                "message": "non - bovine image detected"
            }

        # 5. Check for Other Non-Bovine Animals (Dogs, Cats, Horses, Sheep, Birds)
        if non_bovine_animal_prob_sum > 0.35 and bovine_prob_sum < 0.08 and not is_detected_cow:
            return {
                "is_bovine": False,
                "is_cartoon": False,
                "is_human": False,
                "detected_subject": "Non-Bovine Animal",
                "error": "non - bovine image detected",
                "message": "non - bovine image detected"
            }

        # 6. Positive Photographic Bovine Confirmation
        is_direct_bovine = (
            top_label in self.exact_bovine_classes or
            bovine_prob_sum >= 0.08 or
            (adjacent_livestock_prob_sum >= 0.12 and (bovine_prob_sum >= 0.01 or animal_prob > 0.30)) or
            (is_detected_cow and (bovine_prob_sum > 0.005 or adjacent_livestock_prob_sum > 0.005 or animal_prob > 0.20))
        )

        if is_direct_bovine:
            conf_val = max(top_prob if top_label in self.exact_bovine_classes else 0.0, bovine_prob_sum, 0.65 if is_detected_cow else 0.40)
            return {
                "is_bovine": True,
                "is_cartoon": False,
                "is_human": False,
                "bovine_confidence": round(conf_val * 100, 2),
                "detected_subject": "Bovine (Cattle / Buffalo)",
                "detected_type": "bovine"
            }

        # 7. Close-up fine-grained livestock confirmation for verified animal images
        if (is_detected_cow or animal_prob > 0.25) and fine_grained_conf is not None and fine_grained_conf >= 0.35 and (fine_grained_entropy is None or fine_grained_entropy < 1.95):
            return {
                "is_bovine": True,
                "is_cartoon": False,
                "is_human": False,
                "bovine_confidence": round(fine_grained_conf * 100, 2),
                "detected_subject": "Bovine (Cattle / Buffalo)",
                "detected_type": "bovine"
            }

        # 8. Default Rejection for Out-of-Distribution / Non-Bovine inputs
        return {
            "is_bovine": False,
            "is_cartoon": False,
            "is_human": False,
            "detected_subject": "Non-Bovine Subject",
            "error": "non - bovine image detected",
            "message": "non - bovine image detected"
        }
