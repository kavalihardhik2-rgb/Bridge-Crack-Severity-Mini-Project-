#!/usr/bin/env python3
"""
Create publication-quality plots for the fair comparison experiment.
Also runs additional model evaluations (Supervised ViT-S/16, MoCo v2)
without needing the full evaluate_fair_comparison.py pipeline.
"""

import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import urllib.request

## Encoder parameter counts (backbone only, without classification head)
PARAM_COUNTS = {
    'Supervised':            '25.6M',   # ResNet50
    'Supervised ViT-S/16':   '22M',    # DeiT-Small/16 (supervised)
    'DINO ResNet50':         '23.5M',   # ResNet50 backbone (no fc)
    'DINO ViT-S/16':         '22M',    # ViT-Small/16
    'DINOv2 ViT-S/14':      '22M',    # ViT-Small/14
    'MAE ViT-B/16':          '86M',   # ViT-Base/16
    'I-JEPA':               '632M',    # ViT-Huge/14
    'MoCo v2':               '23.5M',  # ResNet50 backbone (no fc)
    'MoCo v3 ViT-S/16':     '22M',    # ViT-Small/16 (num_heads=12)
}


# =============================================================================
# Dataset Loading (reused from evaluate_fair_comparison.py)
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
# Model Loaders
# =============================================================================

def get_supervised_vit_s16(device):
    """Load supervised ViT-S/16 (DeiT-Small) pretrained on ImageNet-1K."""
    print("Loading supervised ViT-S/16 (DeiT-Small) from timm...")
    import timm

    model = timm.create_model(
        'deit_small_patch16_224',
        pretrained=True,
        num_classes=0,       # Remove classification head
        global_pool='token'  # Use CLS token
    )
    model = model.to(device)
    model.eval()

    class SupervisedViTEncoder(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.feature_dim = 384
            self.is_vit = True
            self.patch_size = 16

        def forward(self, x):
            return self.backbone(x)

    return SupervisedViTEncoder(model), 384


def get_moco_v2_resnet50(device):
    """Load official MoCo v2 ResNet50 pretrained on ImageNet."""
    print("Loading official MoCo v2 ResNet50...")

    cache_dir = os.path.expanduser('~/.cache/simclr')
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(cache_dir, 'moco_v2_800ep.pth.tar')

    if not os.path.exists(checkpoint_path):
        moco_url = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
        print("  Downloading MoCo v2 checkpoint...")
        urllib.request.urlretrieve(moco_url, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    backbone = models.resnet50(weights=None)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.encoder_q.', '').replace('module.', '')
        if new_k.startswith('fc.'):
            continue
        new_state_dict[new_k] = v

    missing, unexpected = backbone.load_state_dict(new_state_dict, strict=False)
    print(f"  Loaded MoCo v2 weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    class MoCoEncoder(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.flatten = nn.Flatten()
            self.feature_dim = 2048
            self.is_vit = False

        def forward(self, x):
            x = self.backbone(x)
            return self.flatten(x)

    encoder = MoCoEncoder(backbone)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder, 2048


# =============================================================================
# Evaluation Functions (reused from evaluate_fair_comparison.py)
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
# Run New Model Evaluations
# =============================================================================

def evaluate_new_models(data_dir, results_path, output_dir, eval_epochs=50):
    """
    Evaluate Supervised ViT-S/16 and MoCo v2, add to existing results JSON.
    Much faster than running the full evaluate_fair_comparison.py.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    percentages = [0.05, 0.1, 0.2, 0.5, 1.0]

    # Load existing results
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Check which models need evaluation
    models_to_run = {}

    if 'Supervised ViT-S/16' not in results['linear']:
        try:
            enc, dim = get_supervised_vit_s16(device)
            models_to_run['Supervised ViT-S/16'] = (enc, dim)
        except Exception as e:
            print(f"  Failed to load Supervised ViT-S/16: {e}")

    if 'MoCo v2' not in results['linear']:
        try:
            enc, dim = get_moco_v2_resnet50(device)
            models_to_run['MoCo v2'] = (enc, dim)
        except Exception as e:
            print(f"  Failed to load MoCo v2: {e}")

    if not models_to_run:
        print("\nAll models already evaluated. Skipping to plots.")
        return results

    # Transforms (same as evaluate_fair_comparison.py)
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
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    full_dataset_train = load_combined_dataset(data_dir, transform=transform_train)
    full_dataset_test = load_combined_dataset(data_dir, transform=transform_test, verbose=False)

    # Train/test split (same seed as evaluate_fair_comparison.py)
    n_samples = len(full_dataset_train)
    indices = list(range(n_samples))
    targets = full_dataset_train.targets

    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=targets, random_state=42
    )

    test_dataset = Subset(full_dataset_test, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    print(f"\nModels to evaluate: {list(models_to_run.keys())}")

    # Initialize results for new models
    for name in models_to_run:
        if name not in results['linear']:
            results['linear'][name] = {}
        if name not in results['finetune']:
            results['finetune'][name] = {}

    # Extract features
    print("\n" + "="*60)
    print("Extracting features...")
    print("="*60)

    all_features = {}
    feature_dims = {}

    for name, (encoder, feature_dim) in models_to_run.items():
        print(f"\n{name}:")
        feature_dims[name] = feature_dim

        train_dataset_for_extract = Subset(full_dataset_test, train_indices)
        train_loader_extract = DataLoader(train_dataset_for_extract, batch_size=64,
                                          shuffle=False, num_workers=4)

        train_feats, train_labels = extract_features(encoder, train_loader_extract, device)
        test_feats, test_labels = extract_features(encoder, test_loader, device)

        all_features[name] = {
            'train_feats': train_feats,
            'train_labels': train_labels,
            'test_feats': test_feats,
            'test_labels': test_labels
        }
        print(f"  Features shape: {train_feats.shape}")

    # Pre-compute nested subsets (same as evaluate_fair_comparison.py)
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
        print(f"  {pct*100:.1f}%: {len(nested_samples[pct])} samples")

    # Evaluate at each percentage
    for pct in percentages:
        print(f"\n{'='*60}")
        print(f"Evaluating with {pct*100:.1f}% labels ({len(nested_samples[pct])} samples)")
        print('='*60)

        sampled_indices = nested_samples[pct]
        actual_train_indices = [train_indices[i] for i in sampled_indices]

        for name in all_features:
            feats = all_features[name]
            feature_dim = feature_dims[name]
            encoder, _ = models_to_run[name]

            # Linear probe
            subset_train_feats = feats['train_feats'][sampled_indices]
            subset_train_labels = feats['train_labels'][sampled_indices]

            linear_acc = train_linear_probe(
                subset_train_feats, subset_train_labels,
                feats['test_feats'], feats['test_labels'],
                feature_dim
            )
            results['linear'][name][str(pct)] = linear_acc
            print(f"  {name} (linear): {linear_acc*100:.2f}%")

            # Fine-tuning
            encoder_copy = copy.deepcopy(encoder)
            is_vit = hasattr(encoder, 'is_vit') and encoder.is_vit

            model = FineTuneModel(encoder_copy, feature_dim, freeze_encoder=False)

            ft_batch_size = 16 if is_vit else 32
            train_subset = Subset(full_dataset_train, actual_train_indices)
            train_loader = DataLoader(train_subset, batch_size=ft_batch_size,
                                      shuffle=True, num_workers=4)

            lr = 1e-5 if is_vit else 1e-4

            finetune_acc = finetune_model(
                model, train_loader, test_loader,
                epochs=eval_epochs, device=device, lr=lr
            )
            results['finetune'][name][str(pct)] = finetune_acc
            print(f"  {name} (finetune): {finetune_acc*100:.2f}%")

            del model, encoder_copy
            torch.cuda.empty_cache()

    # Save updated results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nUpdated results saved to {results_path}")

    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def create_publication_plot(results_path, output_dir):
    """Create publication-quality comparison plot with distinct styles."""

    with open(results_path, 'r') as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Define distinct styles for each method
    styles = {
        'Supervised ResNet50': {
            'color': '#D62728',  # Red
            'marker': 'D',       # Diamond
            'linestyle': '-',    # Solid
            'linewidth': 2.5,
            'markersize': 10,
        },
        'Supervised ViT-S/16': {
            'color': '#8B0000',  # Dark Red
            'marker': 'd',       # Small diamond
            'linestyle': '--',   # Dashed
            'linewidth': 2.5,
            'markersize': 10,
        },
        'DINO ResNet50': {
            'color': '#1F77B4',  # Blue
            'marker': 'o',       # Circle
            'linestyle': '--',   # Dashed
            'linewidth': 2.5,
            'markersize': 10,
        },
        'DINO ViT-S/16': {
            'color': '#2CA02C',  # Green
            'marker': '^',       # Triangle up
            'linestyle': '-.',   # Dash-dot
            'linewidth': 2.5,
            'markersize': 10,
        },
        'DINOv2 ViT-S/14': {
            'color': '#9467BD',  # Purple
            'marker': 's',       # Square
            'linestyle': ':',    # Dotted
            'linewidth': 3.0,    # Thicker for dotted
            'markersize': 10,
        },
        'MAE ViT-B/16': {
            'color': '#FF7F0E',  # Orange
            'marker': 'p',       # Pentagon
            'linestyle': '-',    # Solid
            'linewidth': 2.5,
            'markersize': 11,
        },
        'I-JEPA': {
            'color': '#17BECF',  # Cyan
            'marker': 'h',       # Hexagon
            'linestyle': '--',   # Dashed
            'linewidth': 2.5,
            'markersize': 11,
        },
        'MoCo v2': {
            'color': '#E91E63',  # Pink
            'marker': 'X',       # X marker
            'linestyle': '-.',   # Dash-dot
            'linewidth': 2.5,
            'markersize': 11,
        },
        'MoCo v3 ViT-S/16': {
            'color': '#795548',  # Brown
            'marker': 'v',       # Triangle down
            'linestyle': '-',    # Solid
            'linewidth': 2.5,
            'markersize': 10,
        },
    }

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (mode, title) in zip(axes, [('linear', 'Linear Probe'), ('finetune', 'Fine-tuning')]):
        for name, method_results in results[mode].items():
            if not method_results:
                continue

            style = styles.get(name, {'color': '#333333', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'markersize': 8})

            pcts = sorted([float(p) for p in method_results.keys()])
            accs = [method_results[str(p)] * 100 for p in pcts]

            # Add parameter count to legend label
            params = PARAM_COUNTS.get(name, '?')
            label = f'{name} ({params})'

            ax.plot([p*100 for p in pcts], accs,
                   marker=style['marker'],
                   color=style['color'],
                   linestyle=style['linestyle'],
                   linewidth=style['linewidth'],
                   markersize=style['markersize'],
                   markeredgecolor='white',
                   markeredgewidth=1.5,
                   label=label)

        ax.set_xlabel('Labeled Data (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'{title} Evaluation', fontsize=16, fontweight='bold', pad=15)

        # Better legend
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95,
                 edgecolor='gray', fancybox=True)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Y-axis limits
        ax.set_ylim([88, 97])

        # X-axis ticks
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.set_xticklabels(['5%', '10%', '20%', '50%', '100%'], fontsize=12)

        # Y-axis ticks
        ax.tick_params(axis='y', labelsize=12)

        # Spine styling
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    plt.suptitle('Fair Comparison: Official SSL Models vs Supervised Baselines\n(All ImageNet Pretrained)',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    plt.savefig(os.path.join(output_dir, 'fair_comparison_v2.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(output_dir, 'fair_comparison_v2.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_dir}/fair_comparison_v2.png")
    print(f"Saved: {output_dir}/fair_comparison_v2.pdf")
    plt.close()

    # Also create separate plots for linear and finetune
    for mode, title in [('linear', 'Linear Probe'), ('finetune', 'Fine-tuning')]:
        fig, ax = plt.subplots(figsize=(10, 7))

        for name, method_results in results[mode].items():
            if not method_results:
                continue

            style = styles.get(name, {'color': '#333333', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'markersize': 8})

            pcts = sorted([float(p) for p in method_results.keys()])
            accs = [method_results[str(p)] * 100 for p in pcts]

            # Add parameter count to legend label
            params = PARAM_COUNTS.get(name, '?')
            label = f'{name} ({params})'

            ax.plot([p*100 for p in pcts], accs,
                   marker=style['marker'],
                   color=style['color'],
                   linestyle=style['linestyle'],
                   linewidth=style['linewidth'],
                   markersize=style['markersize'],
                   markeredgecolor='white',
                   markeredgewidth=1.5,
                   label=label)

        ax.set_xlabel('Labeled Data (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'{title}: Official SSL Models vs Supervised\n(ImageNet Pretrained)',
                    fontsize=16, fontweight='bold', pad=15)

        ax.legend(loc='lower right', fontsize=11, framealpha=0.95,
                 edgecolor='gray', fancybox=True)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        if mode == 'linear':
            ax.set_ylim([88, 95])
        else:
            ax.set_ylim([92, 96])

        ax.set_xticks([5, 10, 20, 50, 100])
        ax.set_xticklabels(['5%', '10%', '20%', '50%', '100%'], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f'fair_comparison_{mode}_v2.png'), dpi=300,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.savefig(os.path.join(output_dir, f'fair_comparison_{mode}_v2.pdf'),
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {output_dir}/fair_comparison_{mode}_v2.png")
        plt.close()


def create_results_table_latex(results_path, output_dir):
    """Create a LaTeX-ready results table."""

    with open(results_path, 'r') as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Get all methods and percentages
    all_methods = set()
    all_pcts = set()
    for mode in ['linear', 'finetune']:
        for method, method_results in results[mode].items():
            all_methods.add(method)
            all_pcts.update([float(p) for p in method_results.keys()])

    all_pcts = sorted(all_pcts)
    all_methods = sorted(all_methods)

    # Number of data columns: Params + label fractions
    n_cols = 1 + len(all_pcts)  # params col + pct cols

    # Create LaTeX table
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Fair Comparison: Accuracy (\%) at Different Label Fractions}")
    latex.append(r"\label{tab:fair_comparison}")
    latex.append(r"\begin{tabular}{lc" + "c" * len(all_pcts) + "}")
    latex.append(r"\toprule")

    # Header with Params column
    header = "Method & Params & " + " & ".join([f"{int(p*100)}\\%" for p in all_pcts]) + r" \\"
    latex.append(header)
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{" + str(n_cols + 1) + r"}{c}{\textbf{Linear Probe}} \\")
    latex.append(r"\midrule")

    # Linear probe results
    for method in all_methods:
        if method in results['linear']:
            params = PARAM_COUNTS.get(method, '-')
            row = method.replace('_', r'\_') + " & " + params + " & "
            values = []
            for p in all_pcts:
                if str(p) in results['linear'][method]:
                    val = results['linear'][method][str(p)] * 100
                    values.append(f"{val:.1f}")
                else:
                    values.append("-")
            row += " & ".join(values) + r" \\"
            latex.append(row)

    latex.append(r"\midrule")
    latex.append(r"\multicolumn{" + str(n_cols + 1) + r"}{c}{\textbf{Fine-tuning}} \\")
    latex.append(r"\midrule")

    # Fine-tuning results
    for method in all_methods:
        if method in results['finetune'] and results['finetune'][method]:
            params = PARAM_COUNTS.get(method, '-')
            row = method.replace('_', r'\_') + " & " + params + " & "
            values = []
            for p in all_pcts:
                if str(p) in results['finetune'][method]:
                    val = results['finetune'][method][str(p)] * 100
                    values.append(f"{val:.1f}")
                else:
                    values.append("-")
            row += " & ".join(values) + r" \\"
            latex.append(row)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)

    with open(os.path.join(output_dir, 'results_table.tex'), 'w') as f:
        f.write(latex_str)

    print(f"Saved: {output_dir}/results_table.tex")
    print("\nLaTeX Table:")
    print(latex_str)


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

    # Step 1: Run new model evaluations (Supervised ViT-S/16, MoCo v2)
    print("="*60)
    print("RUNNING NEW MODEL EVALUATIONS")
    print("="*60)
    evaluate_new_models(data_dir, results_path, output_dir, eval_epochs=50)

    # Step 2: Generate publication-quality plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    create_publication_plot(results_path, output_dir)
    create_results_table_latex(results_path, output_dir)

    print("\nDone! All results and plots saved to:", output_dir)
