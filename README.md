# Coffee Leaf Disease & Pest Classification

A Computer Vision classification pipeline for diagnosing Robusta coffee leaf diseases and pests using transfer learning with deep convolutional neural networks (ResNet / EfficientNet).

---

## 1. Dataset & Stratified Split Structure

The dataset combines **4,882 total images** into 4 diagnostic classes using a **70% / 15% / 15% Stratified Split**:

| Class | Total Samples | Train (70%) | Val (15%) | Test (15%) | Primary Origin |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`healthy`** | 1,970 | 1,379 | 296 | 295 | *coffee dataset* & *RoCoLe* |
| **`rust`** | 1,635 | 1,144 | 245 | 246 | *coffee dataset* & *RoCoLe* |
| **`phoma`** | 1,110 | 777 | 166 | 167 | *coffee dataset* |
| **`red_spider_mite`** | 167 | 117 | 25 | 25 | *RoCoLe* |
| **Total** | **4,882** | **3,417** | **732** | **733** | |

---

## 2. Imbalance Handling Strategies

1. **Stratified Splitting**: Guarantees proportional representation across train, validation, and test partitions, preventing zero-support evaluation issues on the minority class.
2. **Inverse Class Frequency Weights**: Cross-Entropy loss is scaled dynamically by $w_c = \frac{N}{C \cdot N_c}$, giving higher penalty to misclassifications on minority classes (e.g. `red_spider_mite`).
3. **Data Augmentations**: Real-time random resized crops, random rotations ($\pm 25^\circ$), horizontal/vertical flips, and color jittering.
4. **Pre-trained Transfer Learning**: Feature extraction and fine-tuning with ImageNet pre-trained backbones (ResNet-18, ResNet-50, EfficientNet-B0).

---

## 3. Project Structure

```text
coffee_leaf_classifier/
├── splits/
│   ├── train.csv                # 3,417 training image paths and labels
│   ├── val.csv                  # 732 validation image paths and labels
│   ├── test.csv                 # 733 independent test image paths and labels
│   └── classes.csv              # Class index to name mapping
├── dataset.py                   # PyTorch Dataset & data augmentation pipelines
├── model.py                     # Transfer learning model architectures
├── train.py                     # Training loop with class-weighted loss and checkpointing
├── evaluate.py                  # Evaluation on test set (Confusion matrix, PR report)
├── checkpoints/                 # Best model weights and training loss/F1 history
└── evaluation_results/          # Evaluation plots and metrics JSON
```

---

## 4. Execution Commands

### Train the Model:
```bash
.venv\Scripts\python train.py --model resnet18 --epochs 15 --batch-size 32 --lr 0.0002
```

### Evaluate on Test Set:
```bash
.venv\Scripts\python evaluate.py
```
