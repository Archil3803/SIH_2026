import os
import json
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_MAPPINGS = {
    "ayrshire cattle": "ayrshire_cattle",
    "brown swiss cattle": "brown_swiss_cattle",
    "holstein friesian cattle": "holstein_friesian_cattle",
    "jersey cattle": "jersey_cattle",
    "red dane cattle": "red_dane_cattle",
    "chhattisgarhi": "chhattisgarhi",
    "jaffarabadi": "jaffarabadi",
    "marathwada": "marathwada",
    "surti": "surti",
    "toda": "toda"
}

CLASS_DISPLAY_NAMES = {
    "ayrshire_cattle": "Ayrshire Cattle",
    "brown_swiss_cattle": "Brown Swiss Cattle",
    "holstein_friesian_cattle": "Holstein Friesian Cattle",
    "jersey_cattle": "Jersey Cattle",
    "red_dane_cattle": "Red Dane Cattle",
    "chhattisgarhi": "Chhattisgarhi Buffalo",
    "jaffarabadi": "Jaffarabadi Buffalo",
    "marathwada": "Marathwada Buffalo",
    "surti": "Surti Buffalo",
    "toda": "Toda Buffalo"
}

def scan_dataset(dataset_dir="dataset"):
    """
    Scans the dataset folder and returns valid image paths mapped to standardized class keys.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples = [] # List of (image_path, class_key)
    classes_found = set()

    base_path = Path(dataset_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")

    for root, dirs, files in os.walk(base_path):
        if not files:
            continue
        
        folder_name = os.path.basename(root).strip().lower()
        class_key = CLASS_MAPPINGS.get(folder_name)
        
        if not class_key:
            # Try fuzzy/partial matching if exact match not found
            for key, val in CLASS_MAPPINGS.items():
                if key in folder_name or folder_name in key:
                    class_key = val
                    break
        
        if not class_key:
            continue

        classes_found.add(class_key)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                file_path = os.path.join(root, f)
                samples.append((file_path, class_key))

    return samples, sorted(list(classes_found))


class CattleDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_key = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback to black image if corrupted
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        label = self.class_to_idx[class_key]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def prepare_dataloaders(dataset_dir="dataset", batch_size=32, val_split=0.2, seed=42, output_meta_dir="models"):
    samples, class_keys = scan_dataset(dataset_dir)
    print(f"Total valid samples indexed: {len(samples)} across {len(class_keys)} classes.")

    class_to_idx = {cls: idx for idx, cls in enumerate(class_keys)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    # Save classes metadata
    os.makedirs(output_meta_dir, exist_ok=True)
    meta_path = os.path.join(output_meta_dir, "classes.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "display_names": {cls: CLASS_DISPLAY_NAMES.get(cls, cls) for cls in class_keys}
        }, f, indent=2)
    print(f"Saved class metadata to {meta_path}")

    # Stratified Train/Val split
    random.seed(seed)
    by_class = {}
    for path, cls in samples:
        by_class.setdefault(cls, []).append((path, cls))

    train_samples = []
    val_samples = []
    for cls, items in by_class.items():
        random.shuffle(items)
        split_point = int(len(items) * (1 - val_split))
        train_samples.extend(items[:split_point])
        val_samples.extend(items[split_point:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"Training samples: {len(train_samples)} | Validation samples: {len(val_samples)}")

    train_transform, val_transform = get_data_transforms()

    train_dataset = CattleDataset(train_samples, class_to_idx, transform=train_transform)
    val_dataset = CattleDataset(val_samples, class_to_idx, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, class_to_idx, idx_to_class


if __name__ == "__main__":
    samples, classes = scan_dataset("dataset")
    print(f"Found {len(samples)} images in {len(classes)} classes:")
    for c in classes:
        print(f" - {c}: {sum(1 for _, k in samples if k == c)} images")
