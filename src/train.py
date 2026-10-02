import time
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import get_dataloaders, compute_class_weights
from model import get_device, build_model, unfreeze_last_n_blocks


PHASE1_EPOCHS = 5     
PHASE2_EPOCHS = 8     
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5      
PATIENCE = 3          

MODEL_SAVE_PATH = "models/best_model.pt"


def run_epoch(model, dataloader, criterion, optimizer, device, train=True):
    
    model.train() if train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    torch.set_grad_enabled(train)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_phase(model, train_loader, val_loader, criterion, optimizer, device,
                 num_epochs, phase_name, history, best_val_acc, best_model_state,
                 patience_counter):
    
    for epoch in range(num_epochs):
        start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["phase"].append(phase_name)

        print(f"[{phase_name}] Epoch {epoch+1}/{num_epochs} "
              f"({elapsed:.0f}s) - "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  -> New best val_acc: {best_val_acc:.4f}, checkpoint saved in memory")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  -> No improvement for {PATIENCE} epochs, stopping {phase_name} early")
                return best_val_acc, best_model_state, patience_counter, True

    return best_val_acc, best_model_state, patience_counter, False


def plot_history(history):
    
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over epochs")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over epochs")
    ax2.legend()

    
    phase1_len = history["phase"].count("phase1")
    if phase1_len < len(epochs) and phase1_len > 0:
        ax1.axvline(x=phase1_len + 0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.axvline(x=phase1_len + 0.5, color="gray", linestyle="--", alpha=0.5, label="Phase 2 starts")

    plt.tight_layout()
    plt.savefig("notebooks/training_curves.png", dpi=150)
    print("Saved training curves to notebooks/training_curves.png")


def main():
    device = get_device()
    print(f"Using device: {device}\n")

    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes\n")

    print("Computing class weights for imbalance handling...")
    class_weights, _ = compute_class_weights()
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("Building model...\n")
    model = build_model(num_classes=num_classes, freeze_base=True)
    model = model.to(device)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "phase": []}
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    
    print("=" * 60)
    print("PHASE 1: Training classifier head (base frozen)")
    print("=" * 60)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR
    )

    best_val_acc, best_model_state, patience_counter, stopped = train_phase(
        model, train_loader, val_loader, criterion, optimizer, device,
        PHASE1_EPOCHS, "phase1", history, best_val_acc, best_model_state, patience_counter
    )

    
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning last blocks (lower learning rate)")
    print("=" * 60)

    model = unfreeze_last_n_blocks(model, n=2)
    patience_counter = 0  

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE2_LR
    )

    best_val_acc, best_model_state, patience_counter, stopped = train_phase(
        model, train_loader, val_loader, criterion, optimizer, device,
        PHASE2_EPOCHS, "phase2", history, best_val_acc, best_model_state, patience_counter
    )

    
    import os
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state_dict": best_model_state,
        "class_names": class_names,
        "best_val_acc": best_val_acc,
    }, MODEL_SAVE_PATH)
    print(f"\nBest model (val_acc={best_val_acc:.4f}) saved to {MODEL_SAVE_PATH}")

    plot_history(history)

    print("\nDone with training. Next: evaluate on the held-out test set (Day 4).")


if __name__ == "__main__":
    main()
