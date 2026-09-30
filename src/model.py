"""
Day 3 - Step 1: Model definition.

Uses EfficientNet-B0 pretrained on ImageNet, with the final classifier
layer replaced to output 38 classes (our plant disease categories).

Why EfficientNet-B0:
  - Strong accuracy-for-its-size tradeoff, well suited to a laptop GPU (MPS)
  - Pretrained ImageNet features (edges, textures, shapes) transfer well
    to leaf images even though ImageNet wasn't trained on plants specifically
  - Fast enough to fine-tune in a few hours on an M2 Pro
"""

import torch
import torch.nn as nn
from torchvision import models


def get_device():
    """
    Auto-detects the best available device:
    MPS (Apple Silicon GPU) > CUDA (NVIDIA GPU) > CPU fallback.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def build_model(num_classes=38, freeze_base=True):
    """
    Loads pretrained EfficientNet-B0 and replaces the final layer.

    freeze_base=True: freezes all pretrained layers except the new
    classifier head. This is what we train FIRST (Phase 1) — it's fast
    since only the small head has trainable parameters, and it prevents
    the pretrained features from being wrecked by large early gradients.

    Later (Phase 2), we unfreeze some deeper layers for fine-tuning
    at a lower learning rate — see train.py.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # EfficientNet-B0's classifier is: Dropout -> Linear(1280, 1000)
    # Replace the Linear layer to output our 38 classes instead of
    # ImageNet's 1000. This new layer is trainable by default.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def unfreeze_last_n_blocks(model, n=2):
    """
    Unfreezes the last n feature blocks of EfficientNet-B0 for fine-tuning
    (Phase 2). EfficientNet-B0's `features` module has 9 blocks (0-8);
    unfreezing the last few lets the model adapt higher-level features
    (which encode shapes/textures closer to "disease spot" concepts)
    to our specific dataset, while keeping early layers (edges, colors -
    already general and useful) frozen.
    """
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

    # Quick smoke test: dummy batch through the model
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"Output shape for dummy batch of 4: {output.shape}")  # expect [4, 38]
