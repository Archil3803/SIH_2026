import os
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class BovineDetector:
    """
    Object detector to localize multiple bovines (cattle & buffaloes) in a scene.
    Uses lightweight SSDLite MobileNetV3 backbone for fast CPU & GPU inference.
    """
    def __init__(self, device=None, score_thresh=0.25, iou_thresh=0.35):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh

        # Load SSDLite MobileNetV3 model
        try:
            self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.categories = self.weights.meta["categories"]
            self.model = ssdlite320_mobilenet_v3_large(weights=self.weights)
            self.model.to(self.device)
            self.model.eval()
            self.transforms = self.weights.transforms()
            self._model_ready = True
        except Exception as e:
            print(f"[WARN] Failed to load SSDLite detection model: {e}")
            self.model = None
            self._model_ready = False

        # Palette for instance bounding box visualization
        self.palette = [
            (16, 185, 129),   # Emerald Green (#10b981)
            (59, 130, 246),   # Sky Blue (#3b82f6)
            (245, 158, 11),   # Amber (#f59e0b)
            (168, 85, 247),   # Purple (#a855f7)
            (236, 72, 153),   # Pink (#ec4899)
            (20, 184, 166),   # Teal (#14b8a6)
            (249, 115, 22),   # Orange (#f97316)
        ]

    def detect_bovines(self, pil_image):
        """
        Detects all cattle/buffalo instances in the image.
        Returns a list of dicts with bounding boxes and normalized coordinates.
        """
        if not self._model_ready or self.model is None:
            return []

        orig_w, orig_h = pil_image.size
        if orig_w == 0 or orig_h == 0:
            return []

        tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(tensor)[0]

        boxes = preds["boxes"].detach().cpu()
        scores = preds["scores"].detach().cpu()
        labels = preds["labels"].detach().cpu()

        # Target COCO classes for bovines & livestock
        # Class 21 is 'cow' (covers cattle, bull, ox, buffalo)
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for box, score, label_idx in zip(boxes, scores, labels):
            score_val = score.item()
            label_name = self.categories[label_idx.item()]

            # We accept 'cow' with standard threshold, or 'sheep'/'horse' if confidence is high
            if label_name == "cow" and score_val >= self.score_thresh:
                filtered_boxes.append(box)
                filtered_scores.append(score_val)
                filtered_labels.append(label_name)
            elif label_name in ("horse", "sheep") and score_val >= max(0.40, self.score_thresh + 0.15):
                # Sometimes buffaloes or cattle are detected as horse/sheep due to coat/horns
                filtered_boxes.append(box)
                filtered_scores.append(score_val)
                filtered_labels.append(label_name)

        if not filtered_boxes:
            return []

        boxes_tensor = torch.stack(filtered_boxes)
        scores_tensor = torch.tensor(filtered_scores)

        # Apply Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, self.iou_thresh)

        detected_instances = []
        for idx in keep_indices:
            box = boxes_tensor[idx].tolist()
            score = scores_tensor[idx].item()
            lbl = filtered_labels[idx.item()]

            x1 = max(0, int(round(box[0])))
            y1 = max(0, int(round(box[1])))
            x2 = min(orig_w, int(round(box[2])))
            y2 = min(orig_h, int(round(box[3])))

            bw = x2 - x1
            bh = y2 - y1

            # Filter out tiny noise boxes (< 5% of image width or height)
            if bw < (orig_w * 0.05) or bh < (orig_h * 0.05):
                continue

            detected_instances.append({
                "box": [x1, y1, x2, y2],
                "box_normalized": [
                    round(x1 / orig_w, 4),
                    round(y1 / orig_h, 4),
                    round(x2 / orig_w, 4),
                    round(y2 / orig_h, 4)
                ],
                "score": round(score, 4),
                "label": lbl,
                "area": bw * bh
            })

        # Sort detections from left to right (x1 ascending) for intuitive numbering (Animal #1, #2...)
        detected_instances.sort(key=lambda d: d["box"][0])
        return detected_instances

    def crop_instance(self, pil_image, box, padding_ratio=0.08):
        """
        Crops an instance bounding box with adaptive context padding.
        """
        orig_w, orig_h = pil_image.size
        x1, y1, x2, y2 = box

        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(orig_w, x2 + pad_x)
        crop_y2 = min(orig_h, y2 + pad_y)

        return pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    def draw_annotated_image(self, pil_image, instances):
        """
        Draws stylish, high-contrast bounding boxes with instance number & breed badge.
        """
        annotated = pil_image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)
        w, h = annotated.size

        font_size = max(14, int(min(w, h) * 0.028))
        try:
            # Try to load a clean TTF font if available
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        for idx, inst in enumerate(instances):
            color = self.palette[idx % len(self.palette)]
            box = inst.get("box", [0, 0, w, h])
            x1, y1, x2, y2 = box

            # Bounding box line thickness proportional to image size
            line_width = max(3, int(min(w, h) * 0.005))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # Header text badge
            animal_num = inst.get("instance_id", idx + 1)
            breed_name = inst.get("breed_name", "Bovine")
            conf = inst.get("confidence_percent", None)
            
            if conf is not None:
                badge_text = f" #{animal_num} {breed_name} ({conf}%) "
            else:
                badge_text = f" #{animal_num} {breed_name} "

            # Calculate text bounding box
            try:
                text_bbox = draw.textbbox((x1, y1), badge_text, font=font)
                tb_w = text_bbox[2] - text_bbox[0]
                tb_h = text_bbox[3] - text_bbox[1] + 8
            except Exception:
                tb_w = len(badge_text) * (font_size * 0.6)
                tb_h = font_size + 8

            # Position badge above box or just inside top if too high
            badge_y1 = max(0, y1 - tb_h - 2) if y1 >= tb_h + 2 else y1 + line_width + 2
            badge_y2 = badge_y1 + tb_h
            badge_x2 = min(w, x1 + tb_w + 10)

            # Draw background capsule for badge
            draw.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=color)
            draw.text((x1 + 4, badge_y1 + 3), badge_text, fill=(255, 255, 255), font=font)

        return annotated


# Global singleton detector instance
_global_detector = None

def get_bovine_detector():
    global _global_detector
    if _global_detector is None:
        _global_detector = BovineDetector()
    return _global_detector
