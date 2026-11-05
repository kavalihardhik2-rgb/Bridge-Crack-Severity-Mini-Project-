#!/usr/bin/env python3
"""
Feature 3: crack segmentation with a U-Net written in plain PyTorch, trained on weak masks.

Run extensions/build_weak_labels.py first. The decision threshold is picked on the validation
split and the numbers reported on the test split. Masks are weak labels from the measurement code,
so the scores show agreement with that code, not with human annotation. No physical crack width
is reported, because the images have no validated scale.

    python extensions/segmentation_unet.py --epochs 15
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extensions.shared import RESULTS_DIR, WORK_DIR, save_json  # noqa: E402

IMG = 256


class SegDS(Dataset):
    def __init__(self, root, split, augment=False):
        self.root, self.split, self.augment = root, split, augment
        self.files = sorted(os.listdir(os.path.join(root, "images", split)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, "images", self.split, f)), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(os.path.join(self.root, "masks", self.split, os.path.splitext(f)[0] + ".png"), 0)
        img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
        if self.augment:                      # same geometry for image and mask
            if np.random.rand() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if np.random.rand() < 0.5:
                img, msk = img[::-1], msk[::-1]
            k = np.random.randint(4)
            img, msk = np.rot90(img, k), np.rot90(msk, k)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0
        x = (x - 0.5) / 0.5
        y = torch.from_numpy(np.ascontiguousarray((msk > 127).astype(np.float32)))[None]
        return x, y


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = block(3, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)
        self.b = block(8 * c, 16 * c)
        self.u4, self.d4 = nn.ConvTranspose2d(16 * c, 8 * c, 2, 2), block(16 * c, 8 * c)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, 2), block(8 * c, 4 * c)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), block(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), block(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.b(F.max_pool2d(e4, 2))
        d4 = self.d4(torch.cat([self.u4(b), e4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def seg_loss(logits, y):
    """Binary cross-entropy plus Dice, with Dice computed over the whole batch."""
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce + dice


def seg_metrics(c):
    tp, fp, fn, _ = c
    return {"IoU": float(tp / max(tp + fp + fn, 1)), "Dice": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(WORK_DIR / "weak"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "segmentation"))
    ap.add_argument("--weights_dir", default=str(WORK_DIR / "models"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = dev == "cuda"
    if not use_cuda:
        print("No GPU found. This will be slow.")
    torch.manual_seed(0); np.random.seed(0)

    dl_tr = DataLoader(SegDS(args.data, "train", True), batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, drop_last=True, pin_memory=use_cuda)
    dl_va = DataLoader(SegDS(args.data, "val"), batch_size=32, num_workers=args.num_workers)
    dl_te = DataLoader(SegDS(args.data, "test"), batch_size=32, num_workers=args.num_workers)
    print("train / val / test images:", len(dl_tr.dataset), len(dl_va.dataset), len(dl_te.dataset))

    @torch.no_grad()
    def eval_counts(model, loader, thrs):
        model.eval()
        c = {t: np.zeros(4) for t in thrs}
        for x, y in loader:
            p = torch.sigmoid(model(x.to(dev)).float())
            y = y.to(dev) > 0.5
            for t in thrs:
                pb = p > t
                tp = (pb & y).sum().item(); fp = (pb & ~y).sum().item(); fn = (~pb & y).sum().item()
                c[t] += [tp, fp, fn, y.numel() - tp - fp - fn]
        return c

    model = UNet(args.base_channels).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.epochs * len(dl_tr))
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    hist, best_dice, best_w = [], -1.0, None

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_cuda):
                logits = model(x)
            loss = seg_loss(logits.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        vm = seg_metrics(eval_counts(model, dl_va, [0.5])[0.5])
        hist.append({"epoch": ep + 1, "train_loss": tot / len(dl_tr), "val_dice": vm["Dice"], "val_iou": vm["IoU"]})
        print(f"epoch {ep+1:>2}/{args.epochs}  loss {hist[-1]['train_loss']:.4f}  "
              f"val Dice {vm['Dice']:.4f}  val IoU {vm['IoU']:.4f}")
        if vm["Dice"] > best_dice:
            best_dice = vm["Dice"]
            best_w = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_w)

    thrs = [0.3, 0.4, 0.5, 0.6, 0.7]
    cv_ = eval_counts(model, dl_va, thrs)
    thr = max(thrs, key=lambda t: seg_metrics(cv_[t])["Dice"])
    test = seg_metrics(eval_counts(model, dl_te, [thr])[thr])
    print(f"\nThreshold chosen on validation: {thr}")
    print("Test metrics (against weak masks):", {k: round(v, 4) for k, v in test.items()})

    @torch.no_grad()
    def per_image(ds):
        model.eval()
        rows, k = [], 0
        for x, y in DataLoader(ds, batch_size=32, num_workers=args.num_workers):
            pb = torch.sigmoid(model(x.to(dev)).float()) > thr
            yb = y.to(dev) > 0.5
            inter = (pb & yb).flatten(1).sum(1).cpu().numpy()
            ps = pb.flatten(1).sum(1).cpu().numpy()
            ys = yb.flatten(1).sum(1).cpu().numpy()
            for j in range(len(x)):
                rows.append((ds.files[k], 2 * inter[j] / max(ps[j] + ys[j], 1), int(ps[j]), int(ys[j])))
                k += 1
        return pd.DataFrame(rows, columns=["file", "dice", "pred_px", "gt_px"])

    pi = per_image(dl_te.dataset)
    neg, posi = pi[pi.gt_px == 0], pi[pi.gt_px > 0].sort_values("dice", ascending=False).reset_index(drop=True)
    fa = int((neg.pred_px > 50).sum()) if len(neg) else 0
    print(f"Cracked test images: mean Dice {posi.dice.mean():.3f}, median {posi.dice.median():.3f}")
    if len(neg):
        print(f"Uncracked test images with more than 50 predicted crack pixels: {fa} of {len(neg)}")

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "threshold": thr, "img_size": IMG,
                "base_channels": args.base_channels, "note": "trained on weak masks, not human annotation"},
               os.path.join(args.weights_dir, "unet_crack.pt"))

    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h.train_loss); ax[0].set_title("training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(h.val_dice, label="val Dice"); ax[1].plot(h.val_iou, label="val IoU"); ax[1].legend()
    ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "training_curves.png"), dpi=150); plt.close(fig)

    pick = list(posi.index[:2]) + list(posi.index[len(posi) // 2 - 1: len(posi) // 2 + 1]) + list(posi.index[-2:])
    fig, axes = plt.subplots(len(pick), 3, figsize=(9, 3 * len(pick)), squeeze=False)
    for r_, i in enumerate(pick):
        f = posi.file[i]
        im = cv2.cvtColor(cv2.imread(os.path.join(args.data, "images", "test", f)), cv2.COLOR_BGR2RGB)
        gm = cv2.imread(os.path.join(args.data, "masks", "test", os.path.splitext(f)[0] + ".png"), 0)
        x, _ = dl_te.dataset[dl_te.dataset.files.index(f)]
        with torch.no_grad():
            pm = (torch.sigmoid(model(x[None].to(dev)).float())[0, 0] > thr).cpu().numpy()
        pm = cv2.resize(pm.astype(np.uint8), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        ov = im.copy(); ov[pm > 0] = (255, 0, 0)
        panels = [(im, "image"), (gm, "weak mask"),
                  (cv2.addWeighted(ov, 0.6, im, 0.4, 0), f"prediction, Dice {posi.dice[i]:.2f}")]
        for c_, (a, t) in enumerate(panels):
            axes[r_, c_].imshow(a, cmap="gray" if a.ndim == 2 else None)
            axes[r_, c_].set_title(t, fontsize=9); axes[r_, c_].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "segmentation_examples.png"), dpi=150); plt.close(fig)

    save_json({"labels": "weak masks from classical crack measurement", "epochs": args.epochs,
               "threshold": thr, "best_val_dice": best_dice, "test": test,
               "cracked_images_dice": {"mean": float(posi.dice.mean()), "median": float(posi.dice.median()),
                                       "n": int(len(posi))},
               "false_alarms": {"count": fa, "of": int(len(neg)), "rule": "more than 50 predicted pixels"},
               "history": hist}, os.path.join(args.out, "segmentation_results.json"))


if __name__ == "__main__":
    main()
