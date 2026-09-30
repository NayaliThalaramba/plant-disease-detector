"""
Day 2 - Step 2 (CSV version): PyTorch Dataset + DataLoader with augmentation,
reading directly from the CSV files produced by split_data.py.

This module is imported by train.py (Day 3) — it's not meant to be run
directly, but has a quick sanity-check block at the bottom for today.
"""

import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/processed"
IMAGE_SIZE = 224  # standard input size for pretrained ImageNet models
BATCH_SIZE = 32

# ImageNet normalization stats — required when using pretrained models,
# since they were trained on images normalized this way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PlantDiseaseDataset(Dataset):
    """
    Custom Dataset that reads (filepath, label) pairs from a CSV file
    instead of expecting images to be physically arranged in folders.
    """

    def __init__(self, csv_path, transform=None):
        self.samples = []  # list of (filepath, label) tuples
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filepath"], int(row["label"])))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_class_names(data_dir=DATA_DIR):
    """Loads the label -> class_name mapping saved by split_data.py."""
    classes_path = f"{data_dir}/classes.csv"
    class_names = {}
    with open(classes_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names[int(row["label"])] = row["class_name"]
    # Return as an ordered list, index-aligned to label integers
    return [class_names[i] for i in range(len(class_names))]


def get_transforms():
    """
    Returns (train_transform, eval_transform).

    Train gets augmentation (random flips, rotation, color jitter) because:
      - Real user photos will have leaves at different angles, in different
        lighting, sometimes slightly off-color from phone cameras.
      - Augmentation makes the model robust to this instead of overfitting
        to "PlantVillage studio conditions" (plain background, consistent
        lighting) which real photos won't match.

    Val/test do NOT get augmentation — we want to evaluate on realistic,
    un-modified data to get a true read on performance.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # leaves can be photographed either way up
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # slight shifts
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=2):
    """Builds train/val/test DataLoaders from the CSV files."""
    train_transform, eval_transform = get_transforms()

    train_dataset = PlantDiseaseDataset(f"{data_dir}/train.csv", transform=train_transform)
    val_dataset = PlantDiseaseDataset(f"{data_dir}/val.csv", transform=eval_transform)
    test_dataset = PlantDiseaseDataset(f"{data_dir}/test.csv", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    class_names = load_class_names(data_dir)

    return train_loader, val_loader, test_loader, class_names


def compute_class_weights(data_dir=DATA_DIR):
    """
    Computes inverse-frequency class weights to help with the class
    imbalance we saw on Day 1 (152 to 5507 images per class).

    These weights get passed into the loss function on Day 3, so the
    model is penalized more for misclassifying rare classes (like
    Potato___healthy) than common ones (like Orange Huanglongbing).
    """
    train_dataset = PlantDiseaseDataset(f"{data_dir}/train.csv")
    labels = np.array([label for _, label in train_dataset.samples])
    class_counts = np.bincount(labels)

    # Inverse frequency, normalized so weights average to 1.0
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)

    class_names = load_class_names(data_dir)
    return torch.tensor(weights, dtype=torch.float32), class_names


def _unnormalize(tensor_img):
    """Reverses ImageNet normalization for visualization purposes."""
    img = tensor_img.numpy().transpose((1, 2, 0))
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = std * img + mean
    return np.clip(img, 0, 1)


def visualize_augmentation(data_dir=DATA_DIR, n_samples=8):
    """
    Quick visual sanity check — shows the SAME image run through the
    training augmentation pipeline multiple times, so you can confirm
    the augmentations look reasonable (not too extreme, not identical).
    """
    train_transform, _ = get_transforms()
    dataset = PlantDiseaseDataset(f"{data_dir}/train.csv", transform=train_transform)
    class_names = load_class_names(data_dir)

    _, label = dataset.samples[0]
    class_name = class_names[label]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Augmentation examples — class: {class_name}", fontsize=12)

    for ax in axes.flat:
        img_tensor, _ = dataset[0]  # re-runs the random transform each call
        img = _unnormalize(img_tensor)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("notebooks/augmentation_examples.png", dpi=150)
    print("Saved augmentation examples to notebooks/augmentation_examples.png")


if __name__ == "__main__":
    # Quick sanity checks you can run today
    print("Building dataloaders...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print(f"Number of classes: {len(class_names)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Pull one batch to confirm shapes are correct
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")  # expect [batch_size, 3, 224, 224]
    print(f"Batch label shape: {labels.shape}")

    print("\nComputing class weights for imbalance handling...")
    weights, classes = compute_class_weights()
    print(f"Weight range: min={weights.min():.3f}, max={weights.max():.3f}")
    print("(Rare classes like Potato___healthy should have HIGH weight, "
          "common ones like Orange Huanglongbing should have LOW weight)")

    print("\nGenerating augmentation visualization...")
    visualize_augmentation()
