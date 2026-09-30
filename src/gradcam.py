"""
Day 4 - Step 2: Grad-CAM visualization.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which
regions of an image most influenced the model's prediction, by looking
at gradients flowing into the last convolutional layer.

This turns "the model predicted X with 94% confidence" into a visual:
a heatmap overlay showing WHERE on the leaf the model is focusing
(ideally: on the actual lesion/spot, not the background or a random
corner - if it's focusing on background, that's a red flag worth
discussing in a write-up).

Run from project root:
    python src/gradcam.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import get_dataloaders, PlantDiseaseDataset, get_transforms, IMAGENET_MEAN, IMAGENET_STD
from model import get_device, build_model

MODEL_PATH = "models/best_model.pt"


def load_trained_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = build_model(num_classes=num_classes, freeze_base=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names


def unnormalize_for_display(tensor_img):
    """Converts a normalized tensor back to a [0,1] numpy image for overlay."""
    img = tensor_img.cpu().numpy().transpose((1, 2, 0))
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = std * img + mean
    return np.clip(img, 0, 1)


def generate_gradcam_grid(model, class_names, device, n_examples=8, save_path="notebooks/gradcam_examples.png"):
    """
    Picks a handful of test images (mix of correct and incorrect predictions
    if possible) and shows the original image next to its Grad-CAM heatmap.
    """
    _, eval_transform = get_transforms()
    test_dataset = PlantDiseaseDataset("data/processed/test.csv", transform=eval_transform)

    # Grad-CAM needs the LAST convolutional layer of the feature extractor.
    # For EfficientNet-B0, that's the last block in `features`.
    target_layers = [model.features[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    # Sample a spread of indices across the dataset for variety
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(test_dataset), size=n_examples, replace=False)

    fig, axes = plt.subplots(2, n_examples, figsize=(3 * n_examples, 6))

    for col, idx in enumerate(indices):
        image_tensor, true_label = test_dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(device)

        # Get model's actual prediction
        with torch.no_grad():
            output = model(input_tensor)
            pred_label = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_label].item()

        # Generate Grad-CAM for the PREDICTED class (what the model actually said)
        targets = [ClassifierOutputTarget(pred_label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        rgb_img = unnormalize_for_display(image_tensor)
        cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        correct = (pred_label == true_label)
        title_color = "green" if correct else "red"

        # Top row: original image
        axes[0, col].imshow(rgb_img)
        axes[0, col].axis("off")
        true_name = class_names[true_label].replace("___", "\n")
        axes[0, col].set_title(f"True:\n{true_name}", fontsize=7)

        # Bottom row: Grad-CAM overlay
        axes[1, col].imshow(cam_overlay)
        axes[1, col].axis("off")
        pred_name = class_names[pred_label].replace("___", "\n")
        axes[1, col].set_title(f"Pred: {pred_name}\n({confidence*100:.1f}%)",
                                 fontsize=7, color=title_color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved Grad-CAM grid to {save_path}")


def generate_single_gradcam(model, class_names, device, image_path, save_path="notebooks/gradcam_single.png"):
    """
    Utility for the API later (Day 5) - runs Grad-CAM on a single
    arbitrary image path (e.g. a photo the user uploads) and saves
    the overlay. Useful to test on your OWN photos, not just test set ones.
    """
    _, eval_transform = get_transforms()

    image = Image.open(image_path).convert("RGB")
    image_tensor = eval_transform(image).unsqueeze(0).to(device)

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    with torch.no_grad():
        output = model(image_tensor)
        pred_label = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_label].item()

    targets = [ClassifierOutputTarget(pred_label)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    rgb_img = unnormalize_for_display(image_tensor[0])
    cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(cam_overlay)
    axes[1].set_title(f"Grad-CAM: {class_names[pred_label]} ({confidence*100:.1f}%)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Prediction: {class_names[pred_label]} ({confidence*100:.1f}% confidence)")
    print(f"Saved Grad-CAM visualization to {save_path}")

    return class_names[pred_label], confidence


def main():
    device = get_device()
    print(f"Using device: {device}\n")

    model, class_names = load_trained_model(device)
    print("Model loaded. Generating Grad-CAM grid from test set...\n")

    generate_gradcam_grid(model, class_names, device, n_examples=8)

    print("\nDone. Open notebooks/gradcam_examples.png to inspect.")
    print("Green titles = correct predictions, red = incorrect.")
    print("Check: does the heatmap focus on the actual lesion/diseased area,")
    print("or on background/irrelevant regions? The former is a good sign,")
    print("the latter suggests the model may be using spurious shortcuts.")


if __name__ == "__main__":
    main()
