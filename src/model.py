import torch
import torch.nn as nn
from torchvision import models


def get_device():
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def build_model(num_classes=38, freeze_base=True):
    
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def unfreeze_last_n_blocks(model, n=2):
    
    total_blocks = len(model.features)
    unfreeze_from = max(0, total_blocks - n)

    for i, block in enumerate(model.features):
        if i >= unfreeze_from:
            for param in block.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Unfroze last {n} blocks. Trainable params: {trainable:,} / {total:,}")

    return model


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    model = build_model(num_classes=38)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Phase 1 (frozen base) - Trainable params: {trainable:,} / {total:,}")

    
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"Output shape for dummy batch of 4: {output.shape}")  
