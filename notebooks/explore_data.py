"""
Day 1: Explore the PlantVillage dataset.
Run this after downloading + unzipping the dataset into data/raw/
"""

import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path("data/raw")  # adjust if your folder structure differs after unzip

# IMPORTANT: PlantVillage ships 3 parallel versions of the same images:
# color / grayscale / segmented. We ALWAYS want "color" — real user photos
# will be in color, and color is often the key disease signal (yellowing,
# rust, brown spots). Training on grayscale would cripple real-world accuracy.
DATASET_VARIANT = "color"

def find_dataset_root(data_dir, variant=DATASET_VARIANT):
    """
    Walks the tree looking for a folder named exactly `variant`
    (e.g. "color") that contains class subfolders full of images.
    Falls back to any folder with images if the named variant isn't found.
    """
    fallback = None
    for root, dirs, files in os.walk(data_dir):
        root_path = Path(root)
        has_class_images = dirs and any(
            f.lower().endswith((".jpg", ".jpeg", ".png"))
            for d in dirs
            for f in os.listdir(root_path / d)[:5]
        )
        if has_class_images:
            if root_path.name.lower() == variant:
                return root_path
            if fallback is None:
                fallback = root_path

    if fallback is not None:
        print(f"WARNING: could not find a folder named '{variant}'. "
              f"Falling back to '{fallback}' — DOUBLE CHECK this is the color version!")
        return fallback

    raise FileNotFoundError("Could not find class folders with images under data/raw")

def explore():
    root = find_dataset_root(DATA_DIR)
    print(f"Dataset root found at: {root}\n")

    classes = sorted([d for d in os.listdir(root) if os.path.isdir(root / d)])
    print(f"Number of classes: {len(classes)}\n")

    # Count images per class
    counts = {}
    for cls in classes:
        cls_path = root / cls
        n_images = len([f for f in os.listdir(cls_path)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        counts[cls] = n_images

    print("Images per class:")
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {n}")

    total = sum(counts.values())
    print(f"\nTotal images: {total}")
    print(f"Min class size: {min(counts.values())}")
    print(f"Max class size: {max(counts.values())}")
    print("(Big gaps between min/max = class imbalance — we'll handle this Day 2/3)")

    # Plot class distribution
    plt.figure(figsize=(14, 8))
    plt.barh(list(counts.keys()), list(counts.values()))
    plt.xlabel("Number of images")
    plt.title("Class distribution in PlantVillage dataset")
    plt.tight_layout()
    plt.savefig("notebooks/class_distribution.png", dpi=150)
    print("\nSaved class distribution plot to notebooks/class_distribution.png")

    # Show a few sample images
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    sample_classes = classes[:8] if len(classes) >= 8 else classes
    for ax, cls in zip(axes.flat, sample_classes):
        cls_path = root / cls
        sample_file = [f for f in os.listdir(cls_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))][0]
        img = Image.open(cls_path / sample_file)
        ax.imshow(img)
        ax.set_title(cls, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("notebooks/sample_images.png", dpi=150)
    print("Saved sample images to notebooks/sample_images.png")

    # Check image sizes (helps decide resize strategy later)
    sizes = []
    for cls in classes[:3]:  # sample from a few classes
        cls_path = root / cls
        files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))][:20]
        for f in files:
            img = Image.open(cls_path / f)
            sizes.append(img.size)

    widths, heights = zip(*sizes)
    print(f"\nSample image sizes (from first 3 classes, 20 each):")
    print(f"  Width  - min: {min(widths)}, max: {max(widths)}, avg: {sum(widths)/len(widths):.0f}")
    print(f"  Height - min: {min(heights)}, max: {max(heights)}, avg: {sum(heights)/len(heights):.0f}")

    return root, classes, counts


if __name__ == "__main__":
    explore()
