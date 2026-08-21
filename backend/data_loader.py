import os
import json
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_MAPPINGS = {
    # Cattle breeds (Indian indigenous & crossbred)
    "gir": "gir",
    "jersey cattle": "jersey_cattle",
    "jersey_cattle": "jersey_cattle",
    "jersey": "jersey_cattle",
    "kankrej": "kankrej",
    "red_sindhi": "red_sindhi",
    "red sindhi": "red_sindhi",
    "sahiwal": "sahiwal",
    # Buffalo breeds
    "chhattisgarhi": "chhattisgarhi",
    "jaffarabadi": "jaffarabadi",
    "marathwada": "marathwada",
    "surti": "surti",
    "toda": "toda",
    # Backward compatibility / aliases
    "ayrshire cattle": "ayrshire_cattle",
    "brown swiss cattle": "brown_swiss_cattle",
    "holstein friesian cattle": "holstein_friesian_cattle",
    "red dane cattle": "red_dane_cattle"
}

CLASS_DISPLAY_NAMES = {
    "chhattisgarhi": "Chhattisgarhi Buffalo",
    "gir": "Gir Cattle",
    "jaffarabadi": "Jaffarabadi Buffalo",
    "jersey_cattle": "Jersey Cattle",
    "kankrej": "Kankrej Cattle",
    "marathwada": "Marathwada Buffalo",
    "red_sindhi": "Red Sindhi Cattle",
    "sahiwal": "Sahiwal Cattle",
    "surti": "Surti Buffalo",
    "toda": "Toda Buffalo",
    # Fallback display names
    "ayrshire_cattle": "Ayrshire Cattle",
    "brown_swiss_cattle": "Brown Swiss Cattle",
    "holstein_friesian_cattle": "Holstein Friesian Cattle",
    "red_dane_cattle": "Red Dane Cattle"
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
        module_dir = Path(__file__).resolve().parent
        candidates = [
            module_dir.parent / dataset_dir,
            module_dir / dataset_dir,
            Path(os.path.join(os.getcwd(), dataset_dir))
        ]
        for c in candidates:
            if c.exists():
                base_path = c
                break
        else:
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


def generate_dataset_csv(dataset_dir="dataset", output_csv="dataset.csv"):
    """
    Generates a structured CSV file for supervised learning with paths and exact breed folder classification columns.
    """
    import csv
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    breed_folders = [
        "Chhattisgarhi", "Gir", "Jaffarabadi", "Jersey cattle", "Kankrej",
        "marathwada", "Red_Sindhi", "Sahiwal", "surti", "toda"
    ]
    breed_to_idx = {name: idx for idx, name in enumerate(sorted(breed_folders, key=lambda x: x.lower()))}

    rows = []
    for root, dirs, files in os.walk(dataset_dir):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                full_path = os.path.join(root, f)
                rel_path = full_path.replace("\\", "/")
                parts = rel_path.split("/")
                category = parts[1] if len(parts) >= 4 else "Bovine"
                folder_name = parts[2] if len(parts) >= 4 else os.path.basename(root)

                matched_breed = None
                for bf in breed_folders:
                    if bf.lower() == folder_name.lower():
                        matched_breed = bf
                        break
                if not matched_breed:
                    matched_breed = folder_name

                row = {
                    "image_path": rel_path,
                    "filename": f,
                    "category": category,
                    "breed": matched_breed,
                    "class_idx": breed_to_idx.get(matched_breed, -1)
                }
                for bf in breed_folders:
                    row[bf] = 1 if bf == matched_breed else 0
                rows.append(row)

    fieldnames = ["image_path", "filename", "category", "breed", "class_idx"] + breed_folders
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Generated supervised CSV dataset at '{output_csv}' with {len(rows)} image records.")
    return output_csv


def load_dataset_from_csv(csv_path="dataset.csv"):
    """
    Loads samples from the generated supervised CSV dataset file.
    Returns: List of (image_path, standardized_class_key)
    """
    import csv
    resolved_csv = csv_path
    if not os.path.exists(resolved_csv):
        module_dir = Path(__file__).resolve().parent
        candidates = [
            module_dir / "data" / "dataset.csv",
            module_dir.parent / "data" / "dataset.csv",
            module_dir.parent / "dataset.csv",
            module_dir / "dataset.csv",
            Path(os.path.join(os.getcwd(), csv_path))
        ]
        for c in candidates:
            if c.exists():
                resolved_csv = str(c)
                break
        else:
            raise FileNotFoundError(f"CSV dataset file '{csv_path}' not found.")

    samples = []
    with open(resolved_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row["image_path"]
            breed_folder = row["breed"]
            class_key = CLASS_MAPPINGS.get(breed_folder.lower(), breed_folder.lower().replace(" ", "_"))
            samples.append((img_path, class_key))

    return samples


if __name__ == "__main__":
    generate_dataset_csv("dataset", "dataset.csv")
    samples, classes = scan_dataset("dataset")
    print(f"Found {len(samples)} images in {len(classes)} classes:")
    for c in classes:
        print(f" - {c}: {sum(1 for _, k in samples if k == c)} images")
