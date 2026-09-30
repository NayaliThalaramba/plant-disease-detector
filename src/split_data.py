"""
Day 2 - Step 1 (CSV version): Split the PlantVillage 'color' dataset into
train / val / test SETS, saved as CSV files containing image paths + labels.

No images are copied — this just records which original file belongs to
which split, so data/raw/ stays completely untouched.

Run from the project root:
    python src/split_data.py
"""

import os
import random
import csv
from pathlib import Path

random.seed(42)  # reproducibility

SOURCE_DIR = Path("data/raw/plantvillage dataset/color")
OUTPUT_DIR = Path("data/processed")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # implied remainder


def split_dataset():
    assert SOURCE_DIR.exists(), f"Source dir not found: {SOURCE_DIR}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in SOURCE_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} classes")

    # class_name -> integer label, alphabetical order (matches how
    # torchvision's ImageFolder assigns labels, for consistency)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    rows = {"train": [], "val": [], "test": []}
    summary = {}

    for cls in classes:
        cls_dir = SOURCE_DIR / cls
        images = [f for f in os.listdir(cls_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # test gets whatever remains, avoids rounding losses

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        label = class_to_idx[cls]

        for f in train_files:
            rows["train"].append((str((cls_dir / f).resolve()), label, cls))
        for f in val_files:
            rows["val"].append((str((cls_dir / f).resolve()), label, cls))
        for f in test_files:
            rows["test"].append((str((cls_dir / f).resolve()), label, cls))

        summary[cls] = (len(train_files), len(val_files), len(test_files))
        print(f"{cls}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # Write CSVs: filepath, label (int), class_name (str, for readability)
    for split in ["train", "val", "test"]:
        random.shuffle(rows[split])  # shuffle so classes aren't grouped in the file
        csv_path = OUTPUT_DIR / f"{split}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "class_name"])
            writer.writerows(rows[split])
        print(f"Wrote {len(rows[split])} rows to {csv_path}")

    # Also save the class list, in order, so dataset.py can rebuild
    # the same label <-> class_name mapping later (e.g. for the API on Day 5)
    classes_path = OUTPUT_DIR / "classes.csv"
    with open(classes_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "class_name"])
        for cls, idx in class_to_idx.items():
            writer.writerow([idx, cls])
    print(f"Wrote class mapping to {classes_path}")

    total_train = sum(v[0] for v in summary.values())
    total_val = sum(v[1] for v in summary.values())
    total_test = sum(v[2] for v in summary.values())
    print(f"\nTotal -> train: {total_train}, val: {total_val}, test: {total_test}")
    print(f"\nDone. No images were copied — only file paths were recorded in {OUTPUT_DIR}/*.csv")


if __name__ == "__main__":
    split_dataset()
