#!/usr/bin/env python3
"""
Evaluate MoCo v3 ViT-S/16 on the bridge defect classification task.
Runs linear probe + fine-tuning at all label fractions and updates
the existing fair_comparison_results.json, then regenerates plots.

MoCo v3 ViT-S/16 architecture:
  - embed_dim=384, depth=12, num_heads=12, patch_size=16
  - Note: num_heads=12 (not 6 as in DeiT-Small)
  - 22M backbone parameters
  - Pretrained for 300 epochs on ImageNet-1K
"""

import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from functools import partial
from tqdm import tqdm
import urllib.request


# =============================================================================
# Dataset Loading (same as create_better_plots.py / evaluate_fair_comparison.py)
# =============================================================================

def load_combined_dataset(data_dir, transform=None, folders=None, verbose=True):
    """Load and combine datasets from multiple folders."""
    if folders is None:
        folders = ['Decks', 'Pavements', 'Walls']

    datasets_list = []
    all_targets = []

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            ds = datasets.ImageFolder(folder_path, transform=transform)
            datasets_list.append(ds)
            all_targets.extend(ds.targets)
            if verbose:
                print(f"  Loaded {folder}: {len(ds)} images")

    if not datasets_list:
        raise ValueError(f"No valid folders found in {data_dir}")

    combined = ConcatDataset(datasets_list)
    if verbose:
        print(f"  Total combined: {len(combined)} images")

    combined.targets = all_targets
    return combined


# =============================================================================
# MoCo v3 ViT-S/16 Loader
# =============================================================================

def get_moco_v3_vit_s16(device):
    """
    Load official MoCo v3 ViT-S/16 pretrained on ImageNet.

    MoCo v3 ViT-S uses embed_dim=384, depth=12, num_heads=12 (not 6).
    Checkpoint: https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar
    """
    from timm.models.vision_transformer import VisionTransformer

    print("Loading official MoCo v3 ViT-S/16...")

    # Download checkpoint
    cache_dir = os.path.expanduser('~/.cache/moco_v3')
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(cache_dir, 'vit-s-300ep.pth.tar')

    if not os.path.exists(checkpoint_path):
        url = "https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"
        print(f"  Downloading MoCo v3 ViT-S checkpoint...")
        urllib.request.urlretrieve(url, checkpoint_path)
        print(f"  Downloaded to {checkpoint_path}")

    # Build ViT-S with MoCo v3's architecture (num_heads=12, not 6)
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,       # No classification head (feature extractor)
        global_pool='token', # Use CLS token
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Strip 'module.base_encoder.' prefix and skip projector head keys
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove DDP prefix
        new_k = k.replace('module.', '')
        # Only keep base_encoder keys, strip prefix
        if new_k.startswith('base_encoder.'):
            new_k = new_k.replace('base_encoder.', '')
            # Skip projector head
            if new_k.startswith('head.'):
                continue
            new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Loaded MoCo v3 weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    class MoCoV3Encoder(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.feature_dim = 384
            self.is_vit = True
            self.patch_size = 16

        def forward(self, x):
            return self.backbone(x)

    encoder = MoCoV3Encoder(model)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder, 384


# =============================================================================
# Evaluation Functions (same as create_better_plots.py)
# =============================================================================

def extract_features(encoder, dataloader, device):
    """Extract features from encoder."""
    encoder.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features", leave=False):
            images = images.to(device)
            feats = encoder(images)
            features.append(feats.cpu())
            labels.append(targets)

    return torch.cat(features), torch.cat(labels)


def train_linear_probe(train_features, train_labels, test_features, test_labels,
                       feature_dim, num_classes=2):
    """Train linear classifier on frozen features using sklearn."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_train = train_features.cpu().numpy()
    y_train = train_labels.cpu().numpy()
    X_test = test_features.cpu().numpy()
    y_test = test_labels.cpu().numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs', random_state=42)
    clf.fit(X_train, y_train)

    return clf.score(X_test, y_test)


class FineTuneModel(nn.Module):
    """Model for fine-tuning with frozen or trainable backbone."""

    def __init__(self, encoder, feature_dim, num_classes=2, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        return self.classifier(features)


def finetune_model(model, train_loader, test_loader, epochs=50, device='cuda', lr=1e-4):
    """Fine-tune the model on labeled data."""
    model = model.to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        if acc > best_acc:
            best_acc = acc

    return best_acc


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_moco_v3(data_dir, results_path, eval_epochs=50):
    """Evaluate MoCo v3 ViT-S/16 and add results to existing JSON."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    percentages = [0.05, 0.1, 0.2, 0.5, 1.0]
    model_name = 'MoCo v3 ViT-S/16'

    # Load existing results
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Check if already evaluated
    if model_name in results['linear'] and len(results['linear'][model_name]) == len(percentages):
        print(f"\n{model_name} already fully evaluated. Skipping.")
        return results

    # Load model
    encoder, feature_dim = get_moco_v3_vit_s16(device)

    # Transforms (same as the other experiments)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    print("\n" + "=" * 60)
    print("Loading dataset...")
    print("=" * 60)
    full_dataset_train = load_combined_dataset(data_dir, transform=transform_train)
    full_dataset_test = load_combined_dataset(data_dir, transform=transform_test, verbose=False)

    # Train/test split (same seed as all other experiments)
    n_samples = len(full_dataset_train)
    indices = list(range(n_samples))
    targets = full_dataset_train.targets

    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=targets, random_state=42
    )

    test_dataset = Subset(full_dataset_test, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")

    # Initialize results for this model
    if model_name not in results['linear']:
        results['linear'][model_name] = {}
    if model_name not in results['finetune']:
        results['finetune'][model_name] = {}

    # Extract features (once, reuse for all linear probes)
    print("\n" + "=" * 60)
    print(f"Extracting features for {model_name}...")
    print("=" * 60)

    train_dataset_for_extract = Subset(full_dataset_test, train_indices)
    train_loader_extract = DataLoader(train_dataset_for_extract, batch_size=64,
                                      shuffle=False, num_workers=4)

    train_feats, train_labels = extract_features(encoder, train_loader_extract, device)
    test_feats, test_labels = extract_features(encoder, test_loader, device)
    print(f"  Features shape: {train_feats.shape}")

    # Pre-compute nested subsets (same logic as other experiments)
    train_targets = [targets[i] for i in train_indices]
    sorted_percentages = sorted(percentages, reverse=True)

    nested_samples = {}
    current_indices = list(range(len(train_indices)))
    current_targets = train_targets.copy()

    for pct in sorted_percentages:
        if pct >= 1.0:
            nested_samples[pct] = list(range(len(train_indices)))
        else:
            target_n = int(len(train_indices) * pct)
            if len(current_indices) > target_n:
                keep_indices, _ = train_test_split(
                    range(len(current_indices)),
                    train_size=target_n,
                    stratify=[current_targets[i] for i in range(len(current_indices))],
                    random_state=42
                )
                current_indices = [current_indices[i] for i in keep_indices]
                current_targets = [current_targets[i] for i in keep_indices]
            nested_samples[pct] = current_indices.copy()

    print("\nNested sampling verification:")
    for pct in sorted(percentages):
        print(f"  {pct * 100:.1f}%: {len(nested_samples[pct])} samples")

    # Evaluate at each percentage
    for pct in percentages:
        pct_key = str(pct)

        # Skip if already computed
        if pct_key in results['linear'].get(model_name, {}) and \
           pct_key in results['finetune'].get(model_name, {}):
            print(f"\n  {pct * 100:.1f}% already evaluated, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating with {pct * 100:.1f}% labels ({len(nested_samples[pct])} samples)")
        print('=' * 60)

        sampled_indices = nested_samples[pct]
        actual_train_indices = [train_indices[i] for i in sampled_indices]

        # Linear probe
        if pct_key not in results['linear'].get(model_name, {}):
            subset_train_feats = train_feats[sampled_indices]
            subset_train_labels = train_labels[sampled_indices]

            linear_acc = train_linear_probe(
                subset_train_feats, subset_train_labels,
                test_feats, test_labels,
                feature_dim
            )
            results['linear'][model_name][pct_key] = linear_acc
            print(f"  {model_name} (linear): {linear_acc * 100:.2f}%")

        # Fine-tuning
        if pct_key not in results['finetune'].get(model_name, {}):
            encoder_copy = copy.deepcopy(encoder)
            model = FineTuneModel(encoder_copy, feature_dim, freeze_encoder=False)

            train_subset = Subset(full_dataset_train, actual_train_indices)
            train_loader = DataLoader(train_subset, batch_size=16,
                                      shuffle=True, num_workers=4)

            finetune_acc = finetune_model(
                model, train_loader, test_loader,
                epochs=eval_epochs, device=device, lr=1e-5  # ViT learning rate
            )
            results['finetune'][model_name][pct_key] = finetune_acc
            print(f"  {model_name} (finetune): {finetune_acc * 100:.2f}%")

            del model, encoder_copy
            torch.cuda.empty_cache()

        # Save after each percentage (in case of interruption)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved.")

    print(f"\n{'=' * 60}")
    print(f"FINAL RESULTS for {model_name}:")
    print(f"{'=' * 60}")
    for pct in percentages:
        lin = results['linear'][model_name].get(str(pct), 0) * 100
        ft = results['finetune'][model_name].get(str(pct), 0) * 100
        print(f"  {pct * 100:.0f}%: Linear={lin:.2f}%, FT={ft:.2f}%")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to SDNET2018 dataset (contains Decks/, Pavements/, Walls/)')
    parser.add_argument('--results_path', type=str, default='results/results.json')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    data_dir = args.data_dir
    results_path = args.results_path
    output_dir = args.output_dir

    print("=" * 60)
    print("EVALUATING MoCo v3 ViT-S/16")
    print("=" * 60)
    results = evaluate_moco_v3(data_dir, results_path, eval_epochs=50)

    # Regenerate plots by importing from create_better_plots
    print("\n" + "=" * 60)
    print("REGENERATING PLOTS")
    print("=" * 60)
    from create_better_plots import create_publication_plot, create_results_table_latex
    create_publication_plot(results_path, output_dir)
    create_results_table_latex(results_path, output_dir)

    print("\nDone! Results updated and plots regenerated.")
