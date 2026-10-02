import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from dataset import get_dataloaders
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

    print(f"Loaded model with val_acc={checkpoint['best_val_acc']:.4f}")
    return model, class_names


def evaluate_on_test(model, test_loader, device):
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, class_names):
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("notebooks/confusion_matrix.png", dpi=150)
    print("Saved confusion matrix to notebooks/confusion_matrix.png")

    return cm, cm_normalized


def find_most_confused_pairs(cm_normalized, class_names, top_n=10):
    
    n = len(class_names)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm_normalized[i, j] > 0:
                pairs.append((cm_normalized[i, j], class_names[i], class_names[j]))

    pairs.sort(reverse=True)
    print(f"\nTop {top_n} most confused pairs (true -> predicted):")
    for rate, true_cls, pred_cls in pairs[:top_n]:
        print(f"  {true_cls} -> {pred_cls}: {rate*100:.1f}% of the time")

    return pairs[:top_n]


def main():
    device = get_device()
    print(f"Using device: {device}\n")

    model, class_names = load_trained_model(device)

    print("Loading test data...")
    _, _, test_loader, _ = get_dataloaders()

    print("Running evaluation on test set...")
    y_true, y_pred = evaluate_on_test(model, test_loader, device)

    test_acc = (y_true == y_pred).mean()
    print(f"\n{'='*60}")
    print(f"TEST SET ACCURACY: {test_acc:.4f}")
    print(f"{'='*60}\n")

    print("Per-class performance:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)

    
    with open("notebooks/classification_report.txt", "w") as f:
        f.write(f"Test set accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    print("Saved full classification report to notebooks/classification_report.txt")

    cm, cm_normalized = plot_confusion_matrix(y_true, y_pred, class_names)
    find_most_confused_pairs(cm_normalized, class_names)


if __name__ == "__main__":
    main()
