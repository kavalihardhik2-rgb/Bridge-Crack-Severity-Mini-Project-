#!/usr/bin/env python3
"""
Build weak boxes and masks for crack localization (Feature 2) and segmentation (Feature 3).

SDNET2018 has no bounding boxes or pixel masks. This script makes them from the classical
measurement code in severity/measure_cracks.py:
  * mask: the crack pixels found by the measurement code
  * boxes: one box per group of nearby crack fragments, written in YOLO format (class 0 = crack)
A share of uncracked images is added with empty labels, so the models also see "no crack, no output".
Patches from one photo always land in the same split (see shared.py).

These are weak labels, not human annotation. Scores measured against them show agreement with the
measurement code. Look at the preview image before training anything on them.

    python extensions/build_weak_labels.py --archive /path/to/archive
"""
import argparse
import os
import shutil
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extensions.shared import DEFAULT_LABELS, RESULTS_DIR, WORK_DIR, assign_split, load_labels  # noqa: E402
from severity.common import list_dataset  # noqa: E402
from severity.measure_cracks import measure_crack  # noqa: E402

MIN_BOX = 6   # skip boxes smaller than this (pixels) in both directions


def boxes_from_mask(mask, merge_px=15, pad=2):
    """One box per cluster of nearby crack fragments."""
    m = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_px, merge_px))
    n, comp = cv2.connectedComponents(cv2.dilate(m, k), connectivity=8)
    H, W = m.shape
    out = []
    for lbl in range(1, n):
        ys, xs = np.where((comp == lbl) & (m > 0))
        if len(xs) == 0:
            continue
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + 1 + pad, W)
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + 1 + pad, H)
        if (x1 - x0) >= MIN_BOX or (y1 - y0) >= MIN_BOX:
            out.append((int(x0), int(y0), int(x1), int(y1)))
    return out


def build_one(job):
    src, stem, split, is_pos, out_dir = job
    img = cv2.imread(src)
    if img is None:
        return (stem, split, -1)
    H, W = img.shape[:2]
    mask, boxes = np.zeros((H, W), np.uint8), []
    if is_pos:
        _, m, _ = measure_crack(img, return_masks=True)
        mask = m.astype(np.uint8) * 255
        boxes = boxes_from_mask(mask)
        if not boxes:                       # nothing usable was measured, skip this image
            return (stem, split, -1)
    shutil.copy(src, os.path.join(out_dir, "images", split, stem + os.path.splitext(src)[1]))
    cv2.imwrite(os.path.join(out_dir, "masks", split, stem + ".png"), mask)
    with open(os.path.join(out_dir, "labels", split, stem + ".txt"), "w") as f:
        for x0, y0, x1, y1 in boxes:
            f.write(f"0 {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0)/W:.6f} {(y1-y0)/H:.6f}\n")
    return (stem, split, len(boxes))


def save_preview(man, out_dir, png, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ex = man[(man.is_crack) & (man.split == "train")]
    ex = ex.sample(min(n, len(ex)), random_state=1)
    fig, axes = plt.subplots(2, len(ex), figsize=(3 * len(ex), 6), squeeze=False)
    for j, s in enumerate(ex.stem):
        f = [x for x in os.listdir(os.path.join(out_dir, "images", "train")) if x.startswith(s)][0]
        im = cv2.cvtColor(cv2.imread(os.path.join(out_dir, "images", "train", f)), cv2.COLOR_BGR2RGB)
        mk = cv2.imread(os.path.join(out_dir, "masks", "train", s + ".png"), 0)
        H, W = mk.shape
        for line in open(os.path.join(out_dir, "labels", "train", s + ".txt")):
            _, cx, cy, bw, bh = map(float, line.split())
            cv2.rectangle(im, (int((cx - bw / 2) * W), int((cy - bh / 2) * H)),
                          (int((cx + bw / 2) * W), int((cy + bh / 2) * H)), (0, 255, 0), 1)
        axes[0, j].imshow(im); axes[0, j].set_title("image and weak box", fontsize=9); axes[0, j].axis("off")
        axes[1, j].imshow(mk, cmap="gray"); axes[1, j].set_title("weak mask", fontsize=9); axes[1, j].axis("off")
    fig.tight_layout(); fig.savefig(png, dpi=150); plt.close(fig)
    print("Preview saved to", png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="folder holding Decks / Pavements / Walls")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--out", default=str(WORK_DIR / "weak"))
    ap.add_argument("--neg_ratio", type=float, default=0.25, help="uncracked images added per cracked image")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--preview", default=str(RESULTS_DIR / "weak_labels_preview.png"))
    args = ap.parse_args()

    lab = load_labels(args.labels)
    paths, targets, _ = list_dataset(args.archive)
    pos = lab[lab.length_px > 0]
    jobs = [(r.path, f"p{r.idx:06d}", assign_split(r.path), True, args.out) for r in pos.itertuples()]
    rng = np.random.RandomState(0)
    neg_idx = rng.choice(np.where(targets == 1)[0], size=int(args.neg_ratio * len(jobs)), replace=False)
    jobs += [(paths[i], f"n{i:06d}", assign_split(paths[i]), False, args.out) for i in neg_idx]

    shutil.rmtree(args.out, ignore_errors=True)
    for sub in ("images", "masks", "labels"):
        for sp in ("train", "val", "test"):
            os.makedirs(os.path.join(args.out, sub, sp), exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x
    with Pool(args.workers) as pool:
        res = list(tqdm(pool.imap_unordered(build_one, jobs, chunksize=32), total=len(jobs), desc="Weak labels"))

    man = pd.DataFrame(res, columns=["stem", "split", "n_boxes"])
    man = man[man.n_boxes >= 0].copy()
    man["is_crack"] = man.stem.str.startswith("p")
    man.to_csv(os.path.join(args.out, "manifest.csv"), index=False)
    print("\nImages per split (columns: no crack / crack)")
    print(man.groupby(["split", "is_crack"]).size().unstack(fill_value=0))
    print("Mean boxes per cracked image:", round(man[man.is_crack].n_boxes.mean(), 2))
    os.makedirs(os.path.dirname(args.preview), exist_ok=True)
    save_preview(man, args.out, args.preview)


if __name__ == "__main__":
    main()
