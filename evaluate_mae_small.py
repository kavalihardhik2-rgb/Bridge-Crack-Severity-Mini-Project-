#!/usr/bin/env python3
"""
MAE Evaluation Script (Fixed)

Fixes:
1. ConcatDataset 'samples' attribute error.
2. Handles missing MAE-Small weights (Defaults to ViT-Base MAE).
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm

# =============================================================================
# Model Loader
# =============================================================================

def get_mae_model(model_name, device):
    """
    Load MAE Model.
    Note: Official MAE weights usually only exist for Base, Large, Huge.
    """
    print(f"\n{'='*40}")
    print(f"Loading {model_name}...")
    print(f"{'='*40}")

    try:
        # Load model from timm
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool='token' 
        )
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print("\nAvailable MAE models in timm usually include:")
        print("  - vit_base_patch16_224.mae (86M params)")
        print("  - vit_large_patch16_224.mae (307M params)")
        print("  - vit_huge_patch14_224.mae (632M params)")
        print("\nIf you strictly need 22M params (Small), consider using DINO:")
        print("  - vit_small_patch16_224.dino")
        return None, None

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded successfully.")
    print(f"  Parameters: {params/1e6:.2f}M")
    
    feature_dim = model.num_features
    model = model.to(device)
    model.eval()

    return model, feature_dim

# =============================================================================
# Helper Classes
# =============================================================================

class FineTuneModel(nn.Module):
    def __init__(self, encoder, feature_dim, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

def load_combined_dataset(data_dir, transform=None):
    """
    Load datasets and explicitely return targets list 
    to fix 'ConcatDataset has no attribute samples' error.
    """
    folders = ['Decks', 'Pavements', 'Walls']
    datasets_list = []
    all_targets = []
    
    print("Loading Data...")
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            ds = datasets.ImageFolder(folder_path, transform=transform)
            datasets_list.append(ds)
            # Accumulate targets for stratification
            all_targets.extend(ds.targets)
            print(f"  Found {folder}: {len(ds)} images")
    
    if not datasets_list:
        raise ValueError(f"No valid folders found in {data_dir}")

    combined = ConcatDataset(datasets_list)
    return combined, all_targets

# =============================================================================
# Evaluation Logic
# =============================================================================

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Extracting features", leave=False):
            imgs = imgs.to(device)
            feats = model(imgs)
            features.append(feats.cpu())
            labels.append(targets)
            
    return torch.cat(features), torch.cat(labels)

def run_linear_probe(train_loader, test_loader, model, device):
    print("  Running Linear Probe...")
    train_feats, train_y = extract_features(model, train_loader, device)
    test_feats, test_y = extract_features(model, test_loader, device)
    
    X_train = train_feats.numpy()
    X_test = test_feats.numpy()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, C=0.01, solver='lbfgs')
    clf.fit(X_train, train_y.numpy())
    
    return clf.score(X_test, test_y.numpy())

def run_finetuning(train_loader, test_loader, base_model, feature_dim, epochs, device):
    print(f"  Running Fine-tuning ({epochs} epochs)...")
    
    import copy
    encoder = copy.deepcopy(base_model)
    model = FineTuneModel(encoder, feature_dim).to(device)
    
    # Optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                total += lbls.size(0)
                correct += (pred == lbls).sum().item()
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            
    return best_acc

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='mae_results_fixed')
    parser.add_argument('--percentages', nargs='+', type=float, 
                        default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument('--epochs', type=int, default=30, help='Fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    
    # Defaulting to ViT-Base MAE because Small MAE does not exist in timm
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224.mae',
                        help='timm model name (e.g., vit_base_patch16_224.mae or vit_small_patch16_224.dino)')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # FIX: Retrieve combined dataset AND the target list
    full_dataset, targets = load_combined_dataset(args.data_dir, transform=transform_train)
    full_dataset_test, _ = load_combined_dataset(args.data_dir, transform=transform_test)

    # Initial Split
    indices = list(range(len(full_dataset)))
    
    # FIX: Use the extracted targets for stratification
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=targets, random_state=42)
    
    test_ds = Subset(full_dataset_test, test_idx)
    test_loader = DataLoader(test_ds, batch_size=64, num_workers=4)

    # 2. Load Model
    mae_model, feature_dim = get_mae_model(args.model_name, device)
    if mae_model is None: return

    # 3. Iterate Percentages
    results = {'linear': {}, 'finetune': {}}
    
    train_targets = [targets[i] for i in train_idx]
    
    for pct in args.percentages:
        print(f"\n--- Evaluating at {pct*100}% Data ---")
        
        # Stratified subset creation
        if pct < 1.0:
            subset_idx, _ = train_test_split(
                train_idx, 
                train_size=pct, 
                stratify=train_targets, 
                random_state=42
            )
        else:
            subset_idx = train_idx

        print(f"  Training Samples: {len(subset_idx)}")

        # Create DataLoaders
        train_ds = Subset(full_dataset, subset_idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        train_ds_eval = Subset(full_dataset_test, subset_idx) 
        train_loader_eval = DataLoader(train_ds_eval, batch_size=64, shuffle=False, num_workers=4)

        # A. Linear Probe
        lin_acc = run_linear_probe(train_loader_eval, test_loader, mae_model, device)
        results['linear'][pct] = lin_acc
        print(f"  >> Linear Probe Accuracy: {lin_acc*100:.2f}%")

        # B. Fine-tuning
        ft_acc = run_finetuning(train_loader, test_loader, mae_model, feature_dim, args.epochs, device)
        results['finetune'][pct] = ft_acc
        print(f"  >> Fine-tuning Accuracy:  {ft_acc*100:.2f}%")

    # 4. Save and Plot
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    plt.figure(figsize=(10, 6))
    x = [p*100 for p in args.percentages]
    y_lin = [results['linear'][p]*100 for p in args.percentages]
    y_ft = [results['finetune'][p]*100 for p in args.percentages]
    
    plt.plot(x, y_lin, 'o-', label=f'{args.model_name} (Linear)', color='blue')
    plt.plot(x, y_ft, 's-', label=f'{args.model_name} (Fine-tune)', color='orange')
    
    plt.xlabel('Percentage of Labels (%)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{args.model_name} Evaluation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, 'performance_plot.png'))
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()