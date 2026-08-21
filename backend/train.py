import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_loader import prepare_dataloaders
from model import build_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = (correct / total) * 100.0
    return val_loss, val_acc


def train(epochs=10, batch_size=32, lr=0.0003, dataset_dir="dataset", output_dir="models", backbone="mobilenet_v3_large"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    train_loader, val_loader, class_to_idx, idx_to_class = prepare_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        val_split=0.2,
        output_meta_dir=output_dir
    )
    num_classes = len(class_to_idx)

    # Build model
    print(f"Building {backbone} model for {num_classes} classes...")
    model = build_model(num_classes=num_classes, pretrained=True, backbone=backbone)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "cattle_classifier.pth")
    history = []

    print("\n" + "="*60)
    print(f"Starting Training: {epochs} Epochs | Batch Size {batch_size} | LR {lr}")
    print("="*60)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_duration = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
                "backbone": backbone,
                "class_to_idx": class_to_idx
            }, best_model_path)
            print(f"  --> Saved new best checkpoint to '{best_model_path}' (Val Acc: {val_acc:.2f}%)")

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"Training Complete in {total_time/60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint saved at: {best_model_path}")
    print("="*60)

    # Save training history log
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_val_acc": best_val_acc,
            "epochs": epochs,
            "history": history
        }, f, indent=2)

    return best_val_acc


if __name__ == "__main__":
    epochs = 6
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
    train(epochs=epochs, batch_size=32, lr=0.0004)
