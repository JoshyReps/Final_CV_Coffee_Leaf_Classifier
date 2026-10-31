"""
Evaluation Script for Coffee Leaf Classifier:
- Loads best checkpoint and evaluates directly on 'final dataset/test' (718 images)
- Generates 4-Class Classification Report & Confusion Matrix
- Computes collapsed Binary metrics (Healthy vs. Unhealthy)
- Generates visual plots for confusion matrix and training curves
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_datasets, CLASSES
from model import build_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "evaluation_results")

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Test Set (4-Class)", fontsize=14, pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot -> {save_path}")

def plot_training_curves(history_file, save_path):
    if not os.path.exists(history_file):
        return
    with open(history_file, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="orange", lw=2)
    axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_f1"], label="Train Macro F1", color="royalblue", lw=2)
    axes[1].plot(epochs, history["val_f1"], label="Val Macro F1", color="forestgreen", lw=2)
    axes[1].set_title("Macro F1-Score vs. Epochs", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curves plot -> {save_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint or lightweight weights
    ckpt_path = os.path.join(SCRIPT_DIR, "checkpoints", "resnet18_coffee_leaf.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_model(model_name="resnet18", num_classes=4, pretrained=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
    else:
        state_dict = checkpoint
        epoch = "best"

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded model weights from {ckpt_path} (epoch: {epoch})")

    # Load test dataset directly from 'final dataset'
    _, test_dataset = get_datasets()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(test_dataset)} test samples across classes: {test_dataset.classes}...")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    class_names = test_dataset.classes
    acc_4cls = accuracy_score(all_labels, all_preds)
    f1_macro_4cls = f1_score(all_labels, all_preds, average="macro")
    cm_4cls = confusion_matrix(all_labels, all_preds)
    report_4cls = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    print("\n" + "="*50)
    print("        4-CLASS TEST EVALUATION REPORT        ")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    print("Overall Test Accuracy: {:.2f}%".format(acc_4cls * 100))
    print("Overall Macro F1-Score: {:.4f}".format(f1_macro_4cls))

    # Binary: Healthy is 0 in sorted classes ['healthy', 'phoma', 'red_spider_mite', 'rust']
    healthy_idx = class_names.index("healthy")
    binary_true = (all_labels != healthy_idx).astype(int)
    binary_pred = (all_preds != healthy_idx).astype(int)

    acc_bin = accuracy_score(binary_true, binary_pred)
    f1_bin = f1_score(binary_true, binary_pred)
    cm_bin = confusion_matrix(binary_true, binary_pred)
    bin_report = classification_report(binary_true, binary_pred, target_names=["Healthy", "Unhealthy"], digits=4)

    print("\n" + "="*50)
    print("   COLLAPSED BINARY EVALUATION (HEALTHY vs UNHEALTHY)   ")
    print("="*50)
    print(bin_report)

    # Save metrics JSON
    results = {
        "model_name": "resnet18",
        "test_samples": len(test_dataset),
        "four_class": {
            "accuracy": float(acc_4cls),
            "macro_f1": float(f1_macro_4cls),
            "classification_report": report_4cls,
            "confusion_matrix": cm_4cls.tolist()
        },
        "binary_collapsed": {
            "accuracy": float(acc_bin),
            "f1_score": float(f1_bin),
            "confusion_matrix": cm_bin.tolist()
        }
    }
    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved test metrics to {metrics_path}")

    plot_confusion_matrix(cm_4cls, class_names, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_training_curves(os.path.join(SCRIPT_DIR, "checkpoints", "training_history.json"),
                         os.path.join(OUTPUT_DIR, "training_curves.png"))

if __name__ == "__main__":
    main()
