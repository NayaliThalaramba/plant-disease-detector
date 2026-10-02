import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/processed"
IMAGE_SIZE = 224  
BATCH_SIZE = 32


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PlantDiseaseDataset(Dataset):
    

    def __init__(self, csv_path, transform=None):
        self.samples = []  
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
    
    classes_path = f"{data_dir}/classes.csv"
    class_names = {}
    with open(classes_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names[int(row["label"])] = row["class_name"]
    
    return [class_names[i] for i in range(len(class_names))]


def get_transforms():
    
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  
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
    
    train_dataset = PlantDiseaseDataset(f"{data_dir}/train.csv")
    labels = np.array([label for _, label in train_dataset.samples])
    class_counts = np.bincount(labels)

    
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)

    class_names = load_class_names(data_dir)
    return torch.tensor(weights, dtype=torch.float32), class_names


def _unnormalize(tensor_img):
    
    img = tensor_img.numpy().transpose((1, 2, 0))
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = std * img + mean
    return np.clip(img, 0, 1)


def visualize_augmentation(data_dir=DATA_DIR, n_samples=8):
    
    train_transform, _ = get_transforms()
    dataset = PlantDiseaseDataset(f"{data_dir}/train.csv", transform=train_transform)
    class_names = load_class_names(data_dir)

    _, label = dataset.samples[0]
    class_name = class_names[label]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Augmentation examples — class: {class_name}", fontsize=12)

    for ax in axes.flat:
        img_tensor, _ = dataset[0]  
        img = _unnormalize(img_tensor)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("notebooks/augmentation_examples.png", dpi=150)
    print("Saved augmentation examples to notebooks/augmentation_examples.png")


if __name__ == "__main__":
    
    print("Building dataloaders...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print(f"Number of classes: {len(class_names)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")  
    print(f"Batch label shape: {labels.shape}")

    print("\nComputing class weights for imbalance handling...")
    weights, classes = compute_class_weights()
    print(f"Weight range: min={weights.min():.3f}, max={weights.max():.3f}")
    print("(Rare classes like Potato___healthy should have HIGH weight, "
          "common ones like Orange Huanglongbing should have LOW weight)")

    print("\nGenerating augmentation visualization...")
    visualize_augmentation()
