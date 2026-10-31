"""
PyTorch Dataset and Data Augmentation Pipeline for Coffee Leaf Classification.
Directly loads from 'final dataset/train' and 'final dataset/test' using ImageFolder with fixed class ordering.
"""

import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

CLASSES = ["healthy", "rust", "phoma", "red_spider_mite"]

class CoffeeLeafFolder(ImageFolder):
    """ImageFolder with deterministic class order matching the trained model."""
    def find_classes(self, directory):
        class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASSES)}
        return CLASSES, class_to_idx

def get_transforms(img_size=224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=25),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    eval_transform = T.Compose([
        T.Resize((int(img_size * 1.14), int(img_size * 1.14))),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, eval_transform

def get_datasets(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final dataset")

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_transform, eval_transform = get_transforms(img_size=224)
    train_dataset = CoffeeLeafFolder(train_dir, transform=train_transform)
    test_dataset = CoffeeLeafFolder(test_dir, transform=eval_transform)

    return train_dataset, test_dataset
