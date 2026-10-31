"""
Training Script for Coffee Leaf Classification:
- Uses Transfer Learning (ResNet-18)
- Trains directly on 'final dataset/train' and validates on 'final dataset/test'
- Incorporates Class-Weighted Cross-Entropy Loss to handle imbalance
- Saves best weights (both full checkpoint and clean <50MB weights for GitHub)
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from dataset import get_datasets, CLASSES
from model import build_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def compute_class_weights(dataset, num_classes=4):
    targets = [target for _, target in dataset.samples]
    counts = np.bincount(targets, minlength=num_classes)
    total = len(dataset)
    weights = total / (num_classes * np.maximum(counts, 1.0))
    print(f"Class counts in training set: {dict(zip(dataset.classes, counts))}")
    print(f"Computed class weights: {dict(zip(dataset.classes, np.round(weights, 3)))}")
    return torch.tensor(weights, dtype=torch.float32)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1

def main():
    parser = argparse.ArgumentParser(description="Train Coffee Leaf Classifier")
    parser.add_argument("--model", type=str, default="resnet18", help="Architecture (resnet18, resnet50, efficientnet_b0)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--use-class-weights", action="store_true", default=True, help="Apply inverse class weights")
    parser.add_argument("--save-dir", type=str, default=os.path.join(SCRIPT_DIR, "checkpoints"), help="Directory to save checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load datasets directly from 'final dataset'
    train_dataset, test_dataset = get_datasets()
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    if args.use_class_weights:
        weights = compute_class_weights(train_dataset, num_classes=len(train_dataset.classes)).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    model = build_model(model_name=args.model, num_classes=len(train_dataset.classes), pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    history = {"train_loss": [], "train_acc": [], "train_f1": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print("\n--- Starting Training ---")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | Macro F1: {train_f1:.4f} || "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save full checkpoint
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
            }, ckpt_path)
            # Save lightweight weights (<50MB) for GitHub
            clean_weights_path = os.path.join(args.save_dir, "resnet18_coffee_leaf.pth")
            torch.save(model.state_dict(), clean_weights_path)
            print(f"  --> Saved new best model to {clean_weights_path} (Val Macro F1: {val_f1:.4f})")

    history_path = os.path.join(args.save_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete! History saved to {history_path}")

if __name__ == "__main__":
    main()
