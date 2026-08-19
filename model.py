import os
import json
import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=10, pretrained=True, backbone="mobilenet_v3_large"):
    """
    Builds a transfer learning model for Cattle & Buffalo Breed Classification.
    """
    if backbone == "mobilenet_v3_large":
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            model = models.mobilenet_v3_large(weights=weights)
        else:
            model = models.mobilenet_v3_large(weights=None)
            
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
    elif backbone == "resnet18":
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(weights=None)
            
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model


def load_trained_model(model_path="models/cattle_classifier.pth", num_classes=10, device="cpu", backbone="mobilenet_v3_large"):
    """
    Loads a saved checkpoint if available, or instantiates a fresh backbone model.
    """
    model = build_model(num_classes=num_classes, pretrained=False, backbone=backbone)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        # Check if saved format is state_dict directly or wrapped checkpoint
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
        elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        print(f"Loaded trained model weights from '{model_path}' successfully.")
    else:
        print(f"Model checkpoint '{model_path}' not found. Initialized with pretrained backbone.")
        model = build_model(num_classes=num_classes, pretrained=True, backbone=backbone)

    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    m = build_model(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print(f"Model created. Output shape: {out.shape}")
