#!/usr/bin/env python3
"""
FAIR Comparison: Official SSL Models vs Supervised Baseline

This script performs a FAIR comparison between SSL and supervised learning:

1. Uses OFFICIAL ImageNet-pretrained SSL models from torch hub / official releases
2. Compares against ImageNet-pretrained supervised baseline (SAME pretraining data!)
3. Includes visualizations: t-SNE embeddings, GradCAM (ResNet), Attention (ViT)

Available Official Models:
- DINO ResNet50: torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
- DINO ViT-S/16: torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
- DINOv2 ViT-S/14: torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
- I-JEPA ViT-H/14: Official Meta checkpoint (downloaded)
- Supervised ResNet50/18: torchvision.models (ImageNet pretrained)

All models are pretrained on ImageNet, making this a FAIR comparison.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import cv2
import urllib.request


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
# Official SSL Model Loaders
# =============================================================================

def get_official_dino_resnet50(device):
    """Load official DINO ResNet50 from torch hub."""
    print("Loading official DINO ResNet50 from torch hub...")
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    model = model.to(device)
    model.eval()

    class DINOResNet50Encoder(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.feature_dim = 2048
            self.is_vit = False

        def forward(self, x):
            return self.backbone(x)

    return DINOResNet50Encoder(model), 2048


def get_official_dino_vits16(device):
    """Load official DINO ViT-S/16 from torch hub."""
    print("Loading official DINO ViT-S/16 from torch hub...")
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model = model.to(device)
    model.eval()

    class DINOViTEncoder(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.feature_dim = 384
            self.is_vit = True
            self.patch_size = 16

        def forward(self, x):
            return self.backbone(x)

        def get_attention(self, x):
            """Get attention maps from DINO ViT."""
            # DINO ViT has a get_last_selfattention method
            return self.backbone.get_last_selfattention(x)

    return DINOViTEncoder(model), 384


def get_official_dinov2_vits14(device):
    """Load official DINOv2 ViT-S/14 from torch hub."""
    print("Loading official DINOv2 ViT-S/14 from torch hub...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device)
    model.eval()

    class DINOv2Encoder(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.feature_dim = 384
            self.is_vit = True
            self.patch_size = 14

        def forward(self, x):
            return self.backbone(x)

        def get_attention(self, x):
            """Get attention maps from DINOv2."""
            # DINOv2 uses get_intermediate_layers with return_class_token
            # We need to extract attention differently
            return self.backbone.get_intermediate_layers(x, n=1, return_class_token=True)

    return DINOv2Encoder(model), 384


def get_official_mae_vitb16(device):
    """
    Load official MAE ViT-B/16 using timm which has proper MAE support.
    """
    print("Loading official MAE ViT-B/16...")

    try:
        import timm
    except ImportError:
        print("  MAE requires 'timm' package. Install with: pip install timm")
        return None, None

    # Use timm's pretrained MAE model - this loads weights correctly
    try:
        # timm has official MAE pretrained weights
        model = timm.create_model(
            'vit_base_patch16_224.mae',  # MAE pretrained ViT-B/16
            pretrained=True,
            num_classes=0,  # Remove classification head
            global_pool='token'  # Use CLS token
        )
        print(f"  Loaded MAE pretrained weights from timm")
    except Exception as e:
        print(f"  Failed to load MAE from timm: {e}")
        print("  Trying fallback with local checkpoint...")

        # Fallback: load from local checkpoint
        checkpoint_path = os.path.expanduser('~/.cache/mae/mae_pretrain_vit_base.pth')
        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            mae_url = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
            print(f"  Downloading MAE checkpoint...")
            try:
                urllib.request.urlretrieve(mae_url, checkpoint_path)
            except Exception as e2:
                print(f"  Failed to download: {e2}")
                return None, None

        # Create model and load checkpoint
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # Filter out decoder weights
        encoder_state = {k: v for k, v in state_dict.items()
                        if not k.startswith('decoder') and 'mask_token' not in k}

        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        print(f"  Loaded local checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    class MAEEncoder(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.feature_dim = 768  # ViT-B has 768 dim
            self.is_vit = True
            self.patch_size = 16

        def forward(self, x):
            return self.backbone(x)

    return MAEEncoder(model), 768


def get_official_ijepa(device, checkpoint_path=None):
    """
    Load official I-JEPA model.

    I-JEPA checkpoints can be downloaded from:
    https://github.com/facebookresearch/ijepa

    If no checkpoint_path provided, we'll try to download ViT-H/14 checkpoint.
    """
    print("Loading official I-JEPA...")

    # Check if we have a local checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.expanduser('~/.cache/ijepa/IN1K-vit.h.14-300e.pth.tar')

    if not os.path.exists(checkpoint_path):
        print(f"  I-JEPA checkpoint not found at {checkpoint_path}")
        print("  To use I-JEPA, download from: https://github.com/facebookresearch/ijepa")
        print("  Skipping I-JEPA...")
        return None, None

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # I-JEPA requires timm for ViT-H/14 architecture
    try:
        import timm
    except ImportError:
        print("  I-JEPA requires 'timm' package for ViT-H/14 architecture.")
        print("  Install with: pip install timm")
        print("  Skipping I-JEPA...")
        return None, None

    # Create ViT-H/14 model matching I-JEPA architecture EXACTLY
    # I-JEPA ViT-H/14: embed_dim=1280, depth=32, num_heads=16
    # I-JEPA does NOT use CLS token - it uses average pooling over all patches
    class IJEPAEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            # IMPORTANT: class_token=False to match I-JEPA's architecture (no CLS token)
            # This gives pos_embed shape [1, 256, 1280] matching the checkpoint
            self.backbone = timm.create_model(
                'vit_huge_patch14_224',
                pretrained=False,
                num_classes=0,
                global_pool='avg',   # Average pooling (like I-JEPA)
                class_token=False    # NO CLS token (like I-JEPA)
            )
            self.feature_dim = 1280
            self.is_vit = True
            self.patch_size = 14

        def forward(self, x):
            return self.backbone(x)

    encoder = IJEPAEncoder()

    # Try to load weights
    try:
        if 'encoder' in checkpoint:
            state_dict = checkpoint['encoder']
        elif 'target_encoder' in checkpoint:
            state_dict = checkpoint['target_encoder']
        else:
            state_dict = checkpoint

        # Clean up state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '').replace('backbone.', '')
            # timm uses 'fc_norm' for final layer norm when global_pool='avg'
            # I-JEPA checkpoint uses 'norm'
            if new_k == 'norm.weight':
                new_k = 'fc_norm.weight'
            elif new_k == 'norm.bias':
                new_k = 'fc_norm.bias'
            new_state_dict[new_k] = v

        # Load weights
        missing, unexpected = encoder.backbone.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        print(f"  Successfully loaded I-JEPA checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"  Warning: Could not load I-JEPA weights: {e}")
        print("  Using randomly initialized weights")

    encoder = encoder.to(device)
    encoder.eval()

    return encoder, encoder.feature_dim


def get_supervised_resnet50(device):
    """Load ImageNet-pretrained supervised ResNet50."""
    print("Loading ImageNet-pretrained supervised ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    class SupervisedResNet50Encoder(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            # Remove classification head
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.flatten = nn.Flatten()
            self.feature_dim = 2048
            self.is_vit = False

        def forward(self, x):
            x = self.backbone(x)
            return self.flatten(x)

    encoder = SupervisedResNet50Encoder(model)
    encoder = encoder.to(device)
    encoder.eval()

    return encoder, 2048


def get_supervised_resnet18(device):
    """Load ImageNet-pretrained supervised ResNet18."""
    print("Loading ImageNet-pretrained supervised ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    class SupervisedResNet18Encoder(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.flatten = nn.Flatten()
            self.feature_dim = 512
            self.is_vit = False

        def forward(self, x):
            x = self.backbone(x)
            return self.flatten(x)

    encoder = SupervisedResNet18Encoder(model)
    encoder = encoder.to(device)
    encoder.eval()

    return encoder, 512


def get_official_moco_resnet50(device):
    """
    Load official MoCo v2 ResNet50 pretrained on ImageNet.

    MoCo v2 is a contrastive learning method similar to SimCLR.
    Official checkpoint from Facebook Research.
    """
    print("Loading official MoCo v2 ResNet50...")

    # Try to load from torch hub (some repos provide this)
    try:
        # Try bolts (PyTorch Lightning Bolts has SimCLR)
        from pl_bolts.models.self_supervised import SimCLR
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
        backbone = simclr.encoder
        print("  Loaded SimCLR from PyTorch Lightning Bolts")

        class SimCLREncoder(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                self.feature_dim = 2048
                self.is_vit = False

            def forward(self, x):
                return self.backbone(x)[-1]  # Get the last feature map

        encoder = SimCLREncoder(backbone)
        encoder = encoder.to(device)
        encoder.eval()
        return encoder, 2048

    except Exception as e:
        print(f"  PyTorch Lightning Bolts not available: {e}")

    # Fallback: Create ResNet50 and load converted SimCLR weights
    try:
        # Download converted SimCLR weights from HuggingFace
        cache_dir = os.path.expanduser('~/.cache/simclr')
        os.makedirs(cache_dir, exist_ok=True)
        checkpoint_path = os.path.join(cache_dir, 'simclr_resnet50_imagenet.pth')

        if not os.path.exists(checkpoint_path):
            # Try HuggingFace hub for SimCLR weights
            print(f"  Downloading SimCLR checkpoint...")
            try:
                # Use huggingface_hub to download
                from huggingface_hub import hf_hub_download
                downloaded_path = hf_hub_download(
                    repo_id="facebook/simclr-rn50",
                    filename="simclr_rn50.pth",
                    cache_dir=cache_dir
                )
                checkpoint_path = downloaded_path
            except Exception as e2:
                print(f"  HuggingFace download failed: {e2}")
                # Alternative: Use MoCo v2 as a SimCLR-like baseline (same contrastive learning family)
                print("  Falling back to MoCo v2 (similar contrastive learning method)...")
                moco_url = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
                try:
                    moco_path = os.path.join(cache_dir, 'moco_v2_800ep.pth.tar')
                    if not os.path.exists(moco_path):
                        urllib.request.urlretrieve(moco_url, moco_path)
                    checkpoint_path = moco_path
                    print("  Downloaded MoCo v2 checkpoint (contrastive learning baseline)")
                except Exception as e3:
                    print(f"  MoCo download also failed: {e3}")
                    return None, None

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if 'classy_state_dict' in checkpoint:
            state_dict = checkpoint['classy_state_dict']['base_model']['model']['trunk']
        elif 'state_dict' in checkpoint:
            # MoCo format
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Create ResNet50 backbone
        backbone = models.resnet50(weights=None)

        # Clean up state dict keys and load
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove various prefixes (handle MoCo's 'module.encoder_q.' prefix)
            new_k = k.replace('module.encoder_q.', '')
            new_k = new_k.replace('_feature_blocks.', '')
            new_k = new_k.replace('module.', '')
            new_k = new_k.replace('backbone.', '')
            new_k = new_k.replace('encoder.', '')
            if new_k.startswith('fc.'):  # Skip classifier
                continue
            new_state_dict[new_k] = v

        # Load weights
        missing, unexpected = backbone.load_state_dict(new_state_dict, strict=False)
        print(f"  Loaded contrastive learning weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        class SimCLREncoder(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                # Remove classification head
                self.backbone = nn.Sequential(*list(backbone.children())[:-1])
                self.flatten = nn.Flatten()
                self.feature_dim = 2048
                self.is_vit = False

            def forward(self, x):
                x = self.backbone(x)
                return self.flatten(x)

        encoder = SimCLREncoder(backbone)
        encoder = encoder.to(device)
        encoder.eval()
        return encoder, 2048

    except Exception as e:
        print(f"  Failed to load SimCLR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# =============================================================================
# Evaluation Classes
# =============================================================================

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


# =============================================================================
# Domain-Specific SSL Training (Approach C)
# =============================================================================

class ContrastiveViewGenerator:
    """Generate two augmented views for contrastive learning."""
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if hasattr(self.base, 'dataset'):
            img, label = self.base.dataset[self.base.indices[idx]]
        else:
            img, label = self.base[idx]
        return [self.transform(img), self.transform(img)], label


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


def train_domain_ssl(encoder, feature_dim, train_dataset, ssl_epochs, device,
                     method='simclr', batch_size=64, lr=1e-4):
    """
    Train domain-specific SSL on top of pretrained encoder (Approach C).

    This takes an ImageNet-pretrained encoder and does additional
    self-supervised learning on domain data to adapt the representations.

    Args:
        encoder: ImageNet-pretrained encoder
        feature_dim: Encoder output dimension
        train_dataset: Domain dataset (raw images, no transform)
        ssl_epochs: Number of SSL training epochs
        device: cuda/cpu
        method: 'simclr' or 'dino'
        batch_size: Batch size for SSL training
        lr: Learning rate (lower than training from scratch)

    Returns:
        Adapted encoder
    """
    import torch.nn.functional as F
    import copy

    print(f"\n{'='*60}")
    print(f"Domain-Specific SSL Training ({method.upper()})")
    print(f"  Epochs: {ssl_epochs}")
    print(f"  Starting from ImageNet-pretrained encoder")
    print(f"{'='*60}")

    # Strong augmentations for SSL
    ssl_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create contrastive dataset
    contrastive_dataset = ContrastiveViewGenerator(train_dataset, ssl_transform)
    dataloader = DataLoader(contrastive_dataset, batch_size=batch_size,
                           shuffle=True, num_workers=4, drop_last=True)

    # Clone encoder for training
    encoder = copy.deepcopy(encoder)
    encoder.train()

    # Add projection head
    projection_head = ProjectionHead(feature_dim, hidden_dim=2048, output_dim=128).to(device)

    # Optimizer with lower LR since we're fine-tuning
    optimizer = optim.AdamW([
        {'params': encoder.parameters(), 'lr': lr * 0.1},  # Lower LR for encoder
        {'params': projection_head.parameters(), 'lr': lr}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ssl_epochs)
    temperature = 0.07

    losses = []
    for epoch in range(ssl_epochs):
        epoch_loss = 0
        n_batches = 0

        for (views, _) in tqdm(dataloader, desc=f"Epoch {epoch+1}/{ssl_epochs}", leave=False):
            view1, view2 = views[0].to(device), views[1].to(device)

            # Forward
            z1 = projection_head(encoder(view1))
            z2 = projection_head(encoder(view2))

            # Normalize
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

            # NT-Xent loss (SimCLR)
            batch_size_actual = z1.shape[0]
            sim_12 = torch.mm(z1, z2.t()) / temperature
            sim_11 = torch.mm(z1, z1.t()) / temperature
            sim_22 = torch.mm(z2, z2.t()) / temperature

            mask = torch.eye(batch_size_actual, dtype=torch.bool, device=device)
            sim_11 = sim_11.masked_fill(mask, float('-inf'))
            sim_22 = sim_22.masked_fill(mask, float('-inf'))

            logits_z1 = torch.cat([sim_12, sim_11], dim=1)
            logits_z2 = torch.cat([sim_12.t(), sim_22], dim=1)

            labels = torch.arange(batch_size_actual, device=device)
            loss = (F.cross_entropy(logits_z1, labels) + F.cross_entropy(logits_z2, labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{ssl_epochs}, Loss: {avg_loss:.4f}")

    encoder.eval()
    print(f"  Domain SSL training complete. Final loss: {losses[-1]:.4f}")

    return encoder


# =============================================================================
# Training Functions
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
                       feature_dim, num_classes=2, epochs=100, device='cuda'):
    """
    Train linear classifier on frozen features.
    Uses sklearn LogisticRegression which is the standard approach for linear probing.
    This properly handles different data sizes and converges correctly.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Convert to numpy
    X_train = train_features.cpu().numpy()
    y_train = train_labels.cpu().numpy()
    X_test = test_features.cpu().numpy()
    y_test = test_labels.cpu().numpy()

    # Normalize features (important for linear probe)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Use LogisticRegression with proper regularization
    # max_iter=1000 ensures convergence for all data sizes
    # C=0.1 provides regularization to avoid overfitting
    clf = LogisticRegression(
        max_iter=1000,
        C=0.1,  # Regularization strength (smaller = more regularization)
        solver='lbfgs',
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluate
    acc = clf.score(X_test, y_test)
    return acc


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

        # Evaluate
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
# Visualization Functions
# =============================================================================

def generate_tsne_embeddings(encoders_dict, test_dataset, output_dir, device, n_samples=2000):
    """
    Generate t-SNE embeddings visualization for all methods.
    """
    print("\n" + "="*60)
    print("Generating t-SNE Embeddings Visualization")
    print("="*60)

    n_methods = len(encoders_dict)
    if n_methods == 0:
        print("No encoders to visualize")
        return

    # Create figure
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    class_names = ['Cracked', 'Non-Cracked']
    colors = ['#E74C3C', '#3498DB']  # Red for cracked, Blue for non-cracked

    # Create dataloader
    if len(test_dataset) > n_samples:
        indices = np.random.choice(len(test_dataset), n_samples, replace=False)
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        dataloader = DataLoader(test_dataset, batch_size=64, sampler=sampler, num_workers=4)
    else:
        dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    for ax_idx, (method_name, encoder) in enumerate(encoders_dict.items()):
        print(f"  Extracting features for {method_name}...")
        encoder.eval()

        features = []
        labels = []

        with torch.no_grad():
            for imgs, lbls in tqdm(dataloader, desc=f"  {method_name}", leave=False):
                imgs = imgs.to(device)
                feats = encoder(imgs)
                features.append(feats.cpu().numpy())
                labels.append(lbls.numpy())

        features = np.concatenate(features)
        labels = np.concatenate(labels)

        # t-SNE
        print(f"  Running t-SNE for {method_name}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embeddings_2d = tsne.fit_transform(features)

        # Plot
        ax = axes[ax_idx]
        for class_id in [0, 1]:
            mask = labels == class_id
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=colors[class_id], label=class_names[class_id],
                      alpha=0.6, s=15, edgecolors='white', linewidth=0.2)

        ax.set_title(method_name, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for ax_idx in range(n_methods, len(axes)):
        axes[ax_idx].set_visible(False)

    plt.suptitle('Feature Embeddings Comparison (t-SNE)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embeddings_tsne.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'embeddings_tsne.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_dir}/embeddings_tsne.png")


def get_gradcam_visualization(model, img_tensor, device):
    """
    Get GradCAM visualization for ResNet models.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("  Warning: pytorch_grad_cam not installed. Skipping GradCAM.")
        return None

    model.eval()
    model = model.to(device)

    # Find target layer (last conv layer in ResNet)
    # Handle TempClassifier which wraps encoder
    target_layer = None

    if hasattr(model, 'encoder') and hasattr(model.encoder, 'backbone'):
        # TempClassifier -> encoder -> backbone (Sequential)
        backbone = model.encoder.backbone
        if isinstance(backbone, nn.Sequential):
            # ResNet backbone as Sequential - find layer4
            for i, layer in enumerate(backbone):
                if hasattr(layer, '__iter__'):
                    # This is likely layer4 (contains BasicBlocks)
                    for block in layer:
                        if hasattr(block, 'conv2'):
                            target_layer = block
            # Use the last block found
        else:
            target_layer = backbone.layer4[-1]
    elif hasattr(model, 'backbone'):
        backbone = model.backbone
        if isinstance(backbone, nn.Sequential):
            for layer in backbone:
                if hasattr(layer, '__iter__'):
                    for block in layer:
                        if hasattr(block, 'conv2'):
                            target_layer = block
        else:
            target_layer = backbone.layer4[-1]
    elif hasattr(model, 'layer4'):
        target_layer = model.layer4[-1]

    if target_layer is None:
        print("  Warning: Could not find target layer for GradCAM")
        return None

    cam = GradCAM(model=model, target_layers=[target_layer])

    input_tensor = img_tensor.unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(0)]  # Target "cracked" class

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return grayscale_cam


def get_vit_attention(encoder, img_tensor, device, patch_size=16):
    """
    Get attention visualization for ViT models.
    Returns attention from CLS token to all patches (or average attention for models without CLS).
    Supports: DINO ViT (torch.hub), DINOv2 (torch.hub), MAE (timm), I-JEPA (timm).
    """
    encoder.eval()
    encoder = encoder.to(device)
    img_tensor = img_tensor.unsqueeze(0).to(device) if img_tensor.dim() == 3 else img_tensor.to(device)

    # Check if model has CLS token (I-JEPA doesn't)
    has_cls_token = True
    if hasattr(encoder, 'backbone'):
        has_cls_token = getattr(encoder.backbone, 'class_token', True)
        if hasattr(encoder.backbone, 'cls_token'):
            has_cls_token = encoder.backbone.cls_token is not None

    # Method 1: DINO ViT from torch.hub (has get_last_selfattention)
    if hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'get_last_selfattention'):
        try:
            with torch.no_grad():
                attn = encoder.backbone.get_last_selfattention(img_tensor)

            # attn shape: [B, num_heads, N, N] where N = 1 (CLS) + num_patches
            attn = attn.mean(dim=1)  # Average over heads: [B, N, N]
            cls_attn = attn[0, 0, 1:]  # CLS to patches (exclude CLS itself)

            num_patches = cls_attn.shape[0]
            h = w = int(num_patches ** 0.5)
            attn_map = cls_attn.reshape(h, w).cpu().numpy()
            attn_map = cv2.resize(attn_map, (224, 224))
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            return attn_map
        except Exception as e:
            print(f"  DINO attention extraction failed: {e}")

    # Method 2: timm ViT models (MAE, I-JEPA) - hook into attention
    if hasattr(encoder, 'backbone'):
        try:
            attn_weights = []

            def hook_fn(module, input, output):
                # timm Attention module stores attention in attn attribute after forward
                # We need to capture it during forward
                pass

            # For timm models, we need to modify the attention forward temporarily
            # to capture attention weights
            backbone = encoder.backbone

            # Find the last attention block
            last_block = None
            if hasattr(backbone, 'blocks'):
                last_block = backbone.blocks[-1]

            if last_block is not None and hasattr(last_block, 'attn'):
                attn_module = last_block.attn

                # Store original forward
                original_forward = attn_module.forward

                # Create new forward that captures attention
                captured_attn = [None]

                def forward_with_attn(x):
                    B, N, C = x.shape
                    qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)

                    # Compute attention weights
                    attn = (q @ k.transpose(-2, -1)) * attn_module.scale
                    attn = attn.softmax(dim=-1)
                    captured_attn[0] = attn.detach()  # [B, num_heads, N, N]

                    # Continue with normal forward
                    attn = attn_module.attn_drop(attn)
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = attn_module.proj(x)
                    x = attn_module.proj_drop(x)
                    return x

                # Replace forward temporarily
                attn_module.forward = forward_with_attn

                with torch.no_grad():
                    _ = encoder(img_tensor)

                # Restore original forward
                attn_module.forward = original_forward

                if captured_attn[0] is not None:
                    attn = captured_attn[0]  # [B, num_heads, N, N]
                    attn = attn.mean(dim=1)  # Average over heads: [B, N, N]

                    if has_cls_token:
                        # CLS token attention to patches
                        cls_attn = attn[0, 0, 1:]  # [num_patches]
                    else:
                        # No CLS token (I-JEPA): average attention across all tokens
                        # Each row is attention FROM that token TO all others
                        # Average across all source tokens
                        cls_attn = attn[0].mean(dim=0)  # [N]

                    num_patches = cls_attn.shape[0]
                    h = w = int(num_patches ** 0.5)
                    attn_map = cls_attn.reshape(h, w).cpu().numpy()
                    attn_map = cv2.resize(attn_map, (224, 224))
                    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                    return attn_map

        except Exception as e:
            print(f"  timm attention extraction failed: {e}")
            import traceback
            traceback.print_exc()

    # Method 3: DINOv2 (different API)
    if hasattr(encoder, 'backbone') and hasattr(encoder.backbone, 'get_intermediate_layers'):
        try:
            # DINOv2 doesn't have get_last_selfattention, need to use hooks
            attn_weights = []

            def attn_hook(module, input, output):
                # DINOv2 uses MemEffAttention which doesn't return weights by default
                pass

            # For DINOv2, we use a simpler approach: feature-based saliency
            with torch.no_grad():
                features = encoder.backbone.get_intermediate_layers(img_tensor, n=1, return_class_token=True)
                patch_tokens = features[0][0]  # [B, num_patches, dim]
                cls_token = features[0][1]  # [B, dim]

                # Compute similarity between CLS and patches
                cls_token = cls_token.unsqueeze(1)  # [B, 1, dim]
                similarity = torch.nn.functional.cosine_similarity(patch_tokens, cls_token, dim=-1)  # [B, num_patches]
                cls_attn = similarity[0]  # [num_patches]

                num_patches = cls_attn.shape[0]
                h = w = int(num_patches ** 0.5)
                attn_map = cls_attn.reshape(h, w).cpu().numpy()
                attn_map = cv2.resize(attn_map, (224, 224))
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                return attn_map
        except Exception as e:
            print(f"  DINOv2 feature similarity failed: {e}")

    print(f"  Warning: Could not extract attention for this model")
    return None


def generate_visualization_comparison(encoders_dict, data_dir, output_dir, device, n_samples=4):
    """
    Generate GradCAM (for ResNet) and Attention (for ViT) visualization.
    """
    print("\n" + "="*60)
    print("Generating GradCAM/Attention Visualization")
    print("="*60)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_raw = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = load_combined_dataset(data_dir, transform=transform, verbose=False)
    dataset_raw = load_combined_dataset(data_dir, transform=transform_raw, verbose=False)

    # Find cracked samples (class 0)
    cracked_indices = [i for i in range(len(dataset)) if dataset.targets[i] == 0]
    np.random.seed(42)
    sample_indices = np.random.choice(cracked_indices, min(n_samples, len(cracked_indices)), replace=False)

    n_methods = len(encoders_dict)
    fig, axes = plt.subplots(len(sample_indices), n_methods + 1,
                             figsize=(3 * (n_methods + 1), 3 * len(sample_indices)))

    if len(sample_indices) == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(sample_indices):
        img_tensor, label = dataset[idx]
        img_raw, _ = dataset_raw[idx]
        img_np = img_raw.permute(1, 2, 0).numpy()

        # Original image
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title('Original' if row == 0 else '', fontsize=10)
        axes[row, 0].axis('off')

        for col, (method_name, encoder) in enumerate(encoders_dict.items(), 1):
            is_vit = getattr(encoder, 'is_vit', False)

            if is_vit:
                # Attention for ViT
                print(f"  Generating attention for {method_name}, sample {row+1}...")
                patch_size = getattr(encoder, 'patch_size', 16)
                attn_map = get_vit_attention(encoder, img_tensor, device, patch_size)

                if attn_map is not None:
                    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                    visualization = 0.4 * img_np + 0.6 * heatmap
                    visualization = np.clip(visualization, 0, 1)
                    axes[row, col].imshow(visualization)
                else:
                    axes[row, col].imshow(img_np)
                    axes[row, col].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                       transform=axes[row, col].transAxes, fontsize=12)

                title_suffix = "(Attention)"
            else:
                # GradCAM for ResNet
                print(f"  Generating GradCAM for {method_name}, sample {row+1}...")

                # Create a classification model for GradCAM
                class TempClassifier(nn.Module):
                    def __init__(self, enc):
                        super().__init__()
                        self.encoder = enc
                        self.fc = nn.Linear(enc.feature_dim, 2)

                    def forward(self, x):
                        feat = self.encoder(x)
                        return self.fc(feat)

                temp_model = TempClassifier(encoder).to(device)
                grayscale_cam = get_gradcam_visualization(temp_model, img_tensor, device)

                if grayscale_cam is not None:
                    try:
                        from pytorch_grad_cam.utils.image import show_cam_on_image
                        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                        axes[row, col].imshow(visualization)
                    except:
                        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                        visualization = 0.4 * img_np + 0.6 * heatmap
                        axes[row, col].imshow(np.clip(visualization, 0, 1))
                else:
                    axes[row, col].imshow(img_np)
                    axes[row, col].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                       transform=axes[row, col].transAxes, fontsize=12)

                title_suffix = "(GradCAM)"

            if row == 0:
                axes[row, col].set_title(f'{method_name}\n{title_suffix}', fontsize=9)
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualization_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'visualization_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_dir}/visualization_comparison.png")


# =============================================================================
# Results Visualization
# =============================================================================

def create_comparison_plot(results, percentages, output_dir):
    """Create publication-quality comparison plot."""

    colors = {
        'Supervised': '#E74C3C',
        'DINO ResNet50': '#3498DB',
        'DINO ViT-S/16': '#2980B9',
        'DINOv2 ViT-S/14': '#9B59B6',
        'MAE ViT-B/16': '#F39C12',
        'I-JEPA': '#27AE60',
        'MoCo v2': '#E91E63',  # Pink
        # Domain-adapted models (Approach C)
        'DINO+Domain': '#1ABC9C',
        'SimCLR+Domain': '#16A085',
    }

    markers = {
        'Supervised': 'D',
        'DINO ResNet50': 'o',
        'DINO ViT-S/16': '^',
        'DINOv2 ViT-S/14': 'v',
        'MAE ViT-B/16': 'h',
        'I-JEPA': 'p',
        'MoCo v2': 'X',  # X marker
        'DINO+Domain': 's',
        'SimCLR+Domain': '*',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (mode, title) in zip(axes, [('linear', 'Linear Probe'), ('finetune', 'Fine-tuning')]):
        for name, method_results in results[mode].items():
            if not method_results:
                continue
            pcts = sorted([float(p) for p in method_results.keys()])
            accs = [method_results[str(p)] * 100 for p in pcts]

            ax.plot([p*100 for p in pcts], accs,
                   marker=markers.get(name, 'o'),
                   color=colors.get(name, '#333333'),
                   linewidth=2, markersize=8, label=name)

        ax.set_xlabel('Percentage of Labeled Data (%)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{title} Evaluation', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([85, 100])

    plt.suptitle('FAIR Comparison: Official SSL vs Supervised (All ImageNet Pretrained)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'fair_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fair_comparison.pdf'), bbox_inches='tight')
    print(f"Plot saved to {output_dir}/fair_comparison.png")
    plt.close()


def create_results_table(results, percentages, output_dir):
    """Create results table visualization."""

    methods = list(results['finetune'].keys())
    pcts = sorted(percentages)

    cell_text = []
    for method in methods:
        row = []
        for p in pcts:
            if str(p) in results['finetune'][method]:
                val = results['finetune'][method][str(p)] * 100
                row.append(f"{val:.1f}%")
            else:
                row.append("-")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    col_labels = [f"{int(p*100)}%" for p in pcts]

    table = ax.table(cellText=cell_text,
                     rowLabels=methods,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Style row labels
    for i in range(len(methods)):
        table[(i+1, -1)].set_facecolor('#34495E')
        table[(i+1, -1)].set_text_props(color='white', fontweight='bold')

    # Highlight best in each column
    for j in range(len(pcts)):
        col_values = []
        for i in range(len(methods)):
            try:
                val = float(cell_text[i][j].replace('%', ''))
                col_values.append((val, i))
            except:
                pass
        if col_values:
            best_val, best_idx = max(col_values)
            table[(best_idx+1, j)].set_facecolor('#27AE60')
            table[(best_idx+1, j)].set_text_props(color='white', fontweight='bold')

    plt.title('FAIR Comparison Results (Fine-tuning Accuracy)',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print text table
    print("\n" + "="*80)
    print("RESULTS TABLE (Fine-tuning Accuracy)")
    print("="*80)
    print(f"\n{'Method':<25}" + "".join([f"{int(p*100):>10}%" for p in pcts]))
    print("-"*80)

    for method in methods:
        row = f"{method:<25}"
        for p in pcts:
            if str(p) in results['finetune'][method]:
                row += f"{results['finetune'][method][str(p)]*100:>10.2f}%"
            else:
                row += f"{'N/A':>11}"
        print(row)


# =============================================================================
# Main Experiment
# =============================================================================

def run_fair_comparison(data_dir, output_dir, percentages, eval_epochs, device,
                        ijepa_checkpoint=None, skip_ijepa=False,
                        domain_ssl=False, ssl_epochs=100, batch_size=64):
    """
    Run the FAIR comparison experiment.

    Args:
        domain_ssl: If True, also train domain-adapted SSL models (Approach C)
        ssl_epochs: Epochs for domain-specific SSL training
        batch_size: Batch size for SSL training
    """

    os.makedirs(output_dir, exist_ok=True)

    # Transforms
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

    # Train/test split
    n_samples = len(full_dataset_train)
    indices = list(range(n_samples))
    targets = full_dataset_train.targets

    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=targets, random_state=42
    )

    test_dataset = Subset(full_dataset_test, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")

    # Load all models
    models_to_eval = {}
    encoders_for_vis = {}

    # Supervised baseline (ResNet50 only - the strong baseline)
    try:
        enc, dim = get_supervised_resnet50(device)
        models_to_eval['Supervised'] = (enc, dim)
        encoders_for_vis['Supervised'] = enc
    except Exception as e:
        print(f"  Failed to load Supervised ResNet50: {e}")

    # SSL models
    try:
        enc, dim = get_official_dino_resnet50(device)
        models_to_eval['DINO ResNet50'] = (enc, dim)
        encoders_for_vis['DINO ResNet50'] = enc
    except Exception as e:
        print(f"  Failed to load DINO ResNet50: {e}")

    try:
        enc, dim = get_official_dino_vits16(device)
        models_to_eval['DINO ViT-S/16'] = (enc, dim)
        encoders_for_vis['DINO ViT-S/16'] = enc
    except Exception as e:
        print(f"  Failed to load DINO ViT-S/16: {e}")

    try:
        enc, dim = get_official_dinov2_vits14(device)
        models_to_eval['DINOv2 ViT-S/14'] = (enc, dim)
        encoders_for_vis['DINOv2 ViT-S/14'] = enc
    except Exception as e:
        print(f"  Failed to load DINOv2 ViT-S/14: {e}")

    # MAE
    try:
        enc, dim = get_official_mae_vitb16(device)
        if enc is not None:
            models_to_eval['MAE ViT-B/16'] = (enc, dim)
            encoders_for_vis['MAE ViT-B/16'] = enc
    except Exception as e:
        print(f"  Failed to load MAE: {e}")

    # SimCLR
    try:
        enc, dim = get_official_moco_resnet50(device)
        if enc is not None:
            models_to_eval['MoCo v2'] = (enc, dim)
            encoders_for_vis['MoCo v2'] = enc
    except Exception as e:
        print(f"  Failed to load SimCLR: {e}")

    # I-JEPA
    if not skip_ijepa:
        try:
            enc, dim = get_official_ijepa(device, ijepa_checkpoint)
            if enc is not None:
                models_to_eval['I-JEPA'] = (enc, dim)
                encoders_for_vis['I-JEPA'] = enc
        except Exception as e:
            print(f"  Failed to load I-JEPA: {e}")

    # =================================================================
    # APPROACH C: Domain-Specific SSL Training
    # =================================================================
    if domain_ssl:
        print("\n" + "="*60)
        print("APPROACH C: Domain-Specific SSL Training")
        print("  ImageNet pretrain -> Domain SSL -> Fine-tune")
        print("="*60)

        # Load raw dataset (no transforms) for SSL training
        raw_dataset = load_combined_dataset(data_dir, transform=None, verbose=False)
        raw_train = Subset(raw_dataset, train_indices)

        # Train DINO + Domain SSL
        if 'DINO ViT-S/16' in models_to_eval:
            try:
                print("\nTraining DINO+Domain (starting from DINO ViT-S/16)...")
                base_enc, dim = models_to_eval['DINO ViT-S/16']
                adapted_enc = train_domain_ssl(
                    base_enc, dim, raw_train, ssl_epochs, device,
                    method='simclr', batch_size=batch_size, lr=1e-4
                )
                models_to_eval['DINO+Domain'] = (adapted_enc, dim)
                encoders_for_vis['DINO+Domain'] = adapted_enc
            except Exception as e:
                print(f"  Failed to train DINO+Domain: {e}")

        # Train SimCLR-style on supervised ResNet50
        if 'Supervised' in models_to_eval:
            try:
                print("\nTraining SimCLR+Domain (starting from Supervised ResNet50)...")
                base_enc, dim = models_to_eval['Supervised']
                adapted_enc = train_domain_ssl(
                    base_enc, dim, raw_train, ssl_epochs, device,
                    method='simclr', batch_size=batch_size, lr=1e-4
                )
                models_to_eval['SimCLR+Domain'] = (adapted_enc, dim)
                encoders_for_vis['SimCLR+Domain'] = adapted_enc
            except Exception as e:
                print(f"  Failed to train SimCLR+Domain: {e}")

    results = {
        'linear': {name: {} for name in models_to_eval},
        'finetune': {name: {} for name in models_to_eval}
    }

    # Extract features for linear probe
    print("\n" + "="*60)
    print("Extracting features...")
    print("="*60)

    all_features = {}
    feature_dims = {}

    for name, (encoder, feature_dim) in models_to_eval.items():
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

    # Pre-compute NESTED subsets for all percentages
    # This ensures smaller percentages are always subsets of larger ones
    train_targets = [targets[i] for i in train_indices]
    sorted_percentages = sorted(percentages, reverse=True)  # Largest first

    nested_samples = {}
    current_indices = list(range(len(train_indices)))  # Start with all
    current_targets = train_targets.copy()

    for pct in sorted_percentages:
        if pct >= 1.0:
            nested_samples[pct] = list(range(len(train_indices)))
        else:
            # Calculate how many samples we need
            target_n = int(len(train_indices) * pct)

            if len(current_indices) > target_n:
                # Subsample from current set (maintaining stratification)
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
            encoder, _ = models_to_eval[name]

            # Linear probe
            subset_train_feats = feats['train_feats'][sampled_indices]
            subset_train_labels = feats['train_labels'][sampled_indices]

            linear_acc = train_linear_probe(
                subset_train_feats, subset_train_labels,
                feats['test_feats'], feats['test_labels'],
                feature_dim, epochs=100, device=device
            )
            results['linear'][name][str(pct)] = linear_acc
            print(f"  {name} (linear): {linear_acc*100:.2f}%")

            # Fine-tuning
            import copy

            # Skip fine-tuning for I-JEPA (ViT-H/14 has 632M params, causes OOM)
            # Don't add to results - will be excluded from fine-tuning plot
            if name == 'I-JEPA':
                print(f"  {name} (finetune): SKIPPED (ViT-H too large for GPU, excluded from plot)")
                continue

            encoder_copy = copy.deepcopy(encoder)

            # Fine-tune ALL models (freeze_encoder=False)
            # DINO ViT models work fine with fine-tuning
            is_vit = hasattr(encoder, 'is_vit') and encoder.is_vit
            freeze_encoder = False  # Actually fine-tune all models

            model = FineTuneModel(encoder_copy, feature_dim, freeze_encoder=freeze_encoder)

            # Smaller batch size for ViTs to avoid OOM
            ft_batch_size = 16 if is_vit else 32

            train_subset = Subset(full_dataset_train, actual_train_indices)
            train_loader = DataLoader(train_subset, batch_size=ft_batch_size, shuffle=True, num_workers=4)

            # Lower LR for ViT fine-tuning (ViTs need MUCH lower LR)
            lr = 1e-5 if is_vit else 1e-4

            finetune_acc = finetune_model(
                model, train_loader, test_loader,
                epochs=eval_epochs, device=device, lr=lr
            )
            results['finetune'][name][str(pct)] = finetune_acc
            print(f"  {name} (finetune): {finetune_acc*100:.2f}%")

            # Clean up GPU memory
            del model, encoder_copy
            torch.cuda.empty_cache()

    # Save results
    results_path = os.path.join(output_dir, 'fair_comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate visualizations
    create_comparison_plot(results, percentages, output_dir)
    create_results_table(results, percentages, output_dir)

    # t-SNE embeddings
    generate_tsne_embeddings(encoders_for_vis, test_dataset, output_dir, device, n_samples=2000)

    # GradCAM / Attention visualization
    generate_visualization_comparison(encoders_for_vis, data_dir, output_dir, device, n_samples=4)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='FAIR Comparison: Official SSL vs Supervised',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment
  python evaluate_fair_comparison.py --data_dir /path/to/data --output_dir results

  # Quick test
  python evaluate_fair_comparison.py --data_dir /path/to/data --percentages 0.1 0.5 1.0

  # With I-JEPA checkpoint
  python evaluate_fair_comparison.py --data_dir /path/to/data --ijepa_checkpoint /path/to/ijepa.pth

Note: All models are ImageNet-pretrained for FAIR comparison.
No SSL training epochs parameter because we use OFFICIAL pretrained models.
        """
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--eval_epochs', type=int, default=50,
                       help='Epochs for fine-tuning evaluation')
    parser.add_argument('--percentages', nargs='+', type=float,
                       default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
                       help='Label percentages to evaluate')
    parser.add_argument('--ijepa_checkpoint', type=str, default=None,
                       help='Path to I-JEPA checkpoint (download from Meta)')
    parser.add_argument('--skip_ijepa', action='store_true',
                       help='Skip I-JEPA evaluation')
    # Domain-specific SSL (Approach C)
    parser.add_argument('--domain_ssl', action='store_true',
                       help='Enable domain-specific SSL training (Approach C): '
                            'ImageNet pretrain -> Domain SSL -> Fine-tune')
    parser.add_argument('--ssl_epochs', type=int, default=100,
                       help='Epochs for domain-specific SSL training (only with --domain_ssl)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for SSL training')

    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'fair_comparison_{timestamp}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("FAIR COMPARISON EXPERIMENT")
    print("="*60)
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print(f"Percentages: {args.percentages}")
    print(f"Eval epochs: {args.eval_epochs}")
    if args.domain_ssl:
        print(f"Domain SSL: ENABLED (Approach C)")
        print(f"SSL epochs: {args.ssl_epochs}")
        print(f"Batch size: {args.batch_size}")
    else:
        print(f"Domain SSL: Disabled (using official pretrained only)")
    print("="*60)

    results = run_fair_comparison(
        args.data_dir, args.output_dir, args.percentages, args.eval_epochs, device,
        args.ijepa_checkpoint, args.skip_ijepa,
        args.domain_ssl, args.ssl_epochs, args.batch_size
    )

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.output_dir}/")
    print(f"  - fair_comparison_results.json")
    print(f"  - fair_comparison.png/.pdf (accuracy plot)")
    print(f"  - results_table.png (table)")
    print(f"  - embeddings_tsne.png/.pdf (t-SNE)")
    print(f"  - visualization_comparison.png/.pdf (GradCAM/Attention)")


if __name__ == '__main__':
    main()