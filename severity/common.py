"""
Shared helpers for the severity-assessment extension.

Design goal: stay consistent with the original repo so that results are comparable.
  * Same file ordering as torchvision.ImageFolder over Decks / Pavements / Walls
    (classes sorted alphabetically per folder, files sorted within each class).
  * Same train/test split as evaluate_fair_comparison.py:
        train_test_split(indices, test_size=0.2, stratify=targets, random_state=42)
  * Crack = class 0 (CD / CP / CW), No crack = class 1 (UD / UP / UW).

This module avoids importing torch at import time so the labeling scripts can run
on a machine without PyTorch.
"""
import os
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

SURFACE_FOLDERS = ["Decks", "Pavements", "Walls"]
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
SEVERITY_NAMES = ["Minor", "Moderate", "Severe"]
CRACK_LABEL = 0      # same convention as ImageFolder on SDNET2018 (C* sorts before U*)
NO_CRACK_LABEL = 1
SPLIT_SEED = 42
TEST_SIZE = 0.2


def list_dataset(data_dir: str, folders: List[str] = None):
    """
    Return (paths, targets, surfaces), ordered exactly like the repo's
    ConcatDataset of ImageFolder datasets.

    targets: 0 = cracked, 1 = uncracked (alphabetical class order within each folder).
    surfaces: the folder each image came from (Decks / Pavements / Walls).
    """
    folders = folders or SURFACE_FOLDERS
    paths, targets, surfaces = [], [], []
    for folder in folders:
        root = os.path.join(data_dir, folder)
        if not os.path.isdir(root):
            continue
        classes = sorted(d.name for d in os.scandir(root) if d.is_dir())
        for class_idx, cls in enumerate(classes):
            cls_dir = os.path.join(root, cls)
            for dirpath, _, fnames in sorted(os.walk(cls_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(IMG_EXTENSIONS):
                        paths.append(os.path.join(dirpath, fname))
                        targets.append(class_idx)
                        surfaces.append(folder)
    if not paths:
        raise ValueError(f"No images found under {data_dir} (expected {folders})")
    return paths, np.array(targets), np.array(surfaces)


def repo_split(targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Identical split to evaluate_fair_comparison.py (stratified 80/20, seed 42)."""
    indices = list(range(len(targets)))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, stratify=targets, random_state=SPLIT_SEED
    )
    return np.array(train_idx), np.array(test_idx)


def nested_subset(indices: np.ndarray, labels: np.ndarray, fraction: float, seed: int = SPLIT_SEED):
    """
    Class-stratified *nested* subsets: the 5% subset is contained in the 10% subset, etc.
    Each class is shuffled once with a fixed seed, then the first `fraction` of it is taken.
    At least one sample per class is kept.
    """
    rng = np.random.RandomState(seed)
    chosen = []
    for c in np.unique(labels):
        cls_idx = indices[labels == c]
        order = rng.permutation(len(cls_idx))
        k = max(1, int(round(fraction * len(cls_idx))))
        chosen.append(cls_idx[order[:k]])
    return np.sort(np.concatenate(chosen))


def get_transform():
    """Same eval-time preprocessing as the repo (224x224 resize + ImageNet normalisation)."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_dinov2_vits14(device):
    """Official DINOv2 ViT-S/14 from torch.hub (384-d CLS features), frozen / eval mode."""
    import torch
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, 384
