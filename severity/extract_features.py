#!/usr/bin/env python3
"""
Extract frozen DINOv2 ViT-S/14 features (384-d) for every SDNET2018 image, once.
Both stage 1 (crack / no crack) and stage 2 (severity) are trained on these cached features,
so the GPU is only needed for this step.

Uses the same eval-time preprocessing as the original repo (Resize 224 + ImageNet norm,
no augmentation).

  python severity/extract_features.py --data_dir /path/to/archive
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from severity.common import get_transform, list_dataset, load_dinov2_vits14  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class ImgDS(Dataset):
    """Module-level so DataLoader workers can pickle it on Windows/macOS (spawn)."""

    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output", default=os.path.join(HERE, "features_dinov2_vits14.npz"))
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths, targets, surfaces = list_dataset(args.data_dir)
    print(f"{len(paths)} images; device = {args.device}")
    tf = get_transform()

    loader = DataLoader(ImgDS(paths, tf), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(args.device == "cuda"))
    model, dim = load_dinov2_vits14(args.device)

    feats = []
    with torch.no_grad():
        for x in tqdm(loader, desc="DINOv2 features"):
            feats.append(model(x.to(args.device)).float().cpu().numpy())
    feats = np.concatenate(feats).astype(np.float32)
    assert feats.shape == (len(paths), dim), feats.shape

    np.savez_compressed(args.output, features=feats, targets=targets,
                        surfaces=surfaces, paths=np.array(paths))
    print(f"Saved {feats.shape} features -> {args.output}")


if __name__ == "__main__":
    main()
