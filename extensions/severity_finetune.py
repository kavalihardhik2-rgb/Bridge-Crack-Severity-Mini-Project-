#!/usr/bin/env python3
"""
Feature 1 (C, optional): fine-tune the last blocks of DINOv2 ViT-S/14 for severity and compare
with the frozen-encoder head on the same split and the same test images.

Run extensions/severity_validated.py first so the frozen result is available for the comparison
(if it is missing, a frozen head with C=0.1 is fitted here instead).

    python extensions/severity_finetune.py --epochs 5 --unfreeze 2
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extensions.shared import (DEFAULT_FEATURES, DEFAULT_LABELS, RESULTS_DIR, SEV_NAMES, WORK_DIR,  # noqa: E402
                               fit_head, load_severity_split, report_dict, save_json)

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF_TRAIN = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(), transforms.ColorJitter(0.2, 0.2),
                               transforms.ToTensor(), NORM])
TF_EVAL = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])


class SevDS(Dataset):
    def __init__(self, paths, labels, tf):
        self.p, self.y, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), int(self.y[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(DEFAULT_FEATURES))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--out", default=str(RESULTS_DIR / "severity"))
    ap.add_argument("--weights_dir", default=str(WORK_DIR / "models"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--unfreeze", type=int, default=2, help="number of final transformer blocks to train")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    args = ap.parse_args()

    from sklearn.metrics import f1_score
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = dev == "cuda"
    if not use_cuda:
        print("No GPU found. This will be slow.")
    data = load_severity_split(args.features, args.labels)
    lab, y, tr, va, te = data["lab"], data["y"], data["tr"], data["va"], data["te"]
    paths = lab.path.values

    mk = lambda idx, tf, shuf: DataLoader(SevDS(paths[idx], y[idx], tf), batch_size=args.batch_size if shuf else 64,
                                          shuffle=shuf, num_workers=args.num_workers, drop_last=shuf)
    dl_tr, dl_va, dl_te = mk(tr, TF_TRAIN, True), mk(va, TF_EVAL, False), mk(te, TF_EVAL, False)

    torch.manual_seed(0); np.random.seed(0)
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev)
    for p in backbone.parameters():
        p.requires_grad = False
    for blk in backbone.blocks[-args.unfreeze:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in backbone.norm.parameters():
        p.requires_grad = True
    head = nn.Linear(384, 3).to(dev)

    cnt = np.bincount(y[tr], minlength=3).astype(float)
    w = torch.tensor(len(tr) / (3 * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW([
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": args.lr_backbone},
        {"params": head.parameters(), "lr": args.lr_head}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    @torch.no_grad()
    def predict(loader):
        backbone.eval(); head.eval()
        P, T = [], []
        for xb, yb in loader:
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_cuda):
                lo = head(backbone(xb.to(dev)))
            P.append(torch.softmax(lo.float(), 1).cpu().numpy()); T.append(yb.numpy())
        return np.concatenate(P), np.concatenate(T)

    best_f1, best_state, history = -1.0, None, []
    for ep in range(args.epochs):
        backbone.train(); head.train()
        tot = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_cuda):
                lo = head(backbone(xb))
            loss = loss_fn(lo.float(), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        pv, tv = predict(dl_va)
        f1v = f1_score(tv, pv.argmax(1), labels=[0, 1, 2], average="macro", zero_division=0)
        history.append({"epoch": ep + 1, "train_loss": tot / len(dl_tr), "val_macro_f1": float(f1v)})
        print(f"epoch {ep+1}/{args.epochs}  train loss {tot/len(dl_tr):.4f}  val macro-F1 {f1v:.4f}")
        if f1v > best_f1:
            best_f1 = f1v
            best_state = (copy.deepcopy(backbone.state_dict()), copy.deepcopy(head.state_dict()))

    backbone.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    pt, tt = predict(dl_te)
    rep = report_dict(tt, pt.argmax(1))

    # frozen reference on the same split
    frozen_json = os.path.join(args.out, "severity_validated_results.json")
    if os.path.exists(frozen_json):
        frozen = json.load(open(frozen_json))["test"]
    else:
        sc, clf = fit_head(data["X"][tr], y[tr], 0.1)
        frozen = report_dict(y[te], clf.predict(sc.transform(data["X"][te])))

    print("\nTest results, fine-tuned DINOv2")
    for n, v in rep["per_class"].items():
        print(f"  {n:<9} precision {v['precision']:.3f}  recall {v['recall']:.3f}  F1 {v['f1']:.3f}  n={v['support']}")
    print(f"\nSame split, same test images:")
    print(f"  frozen encoder          macro-F1 {frozen['macro_f1']:.4f}   Severe->Minor {frozen['critical_errors']['severe_as_minor']}")
    print(f"  fine-tuned ({args.unfreeze} blocks)   macro-F1 {rep['macro_f1']:.4f}   Severe->Minor {rep['critical_errors']['severe_as_minor']}")

    os.makedirs(args.weights_dir, exist_ok=True)
    torch.save({"backbone": best_state[0], "head": best_state[1], "classes": SEV_NAMES},
               os.path.join(args.weights_dir, "severity_finetuned.pt"))
    save_json({"epochs": args.epochs, "unfrozen_blocks": args.unfreeze, "history": history,
               "test": rep, "frozen_reference": frozen},
              os.path.join(args.out, "severity_finetuned_results.json"))


if __name__ == "__main__":
    main()
