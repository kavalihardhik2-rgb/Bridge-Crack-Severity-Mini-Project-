#!/usr/bin/env python3
"""
Feature 2: crack localization with a small YOLO model, trained on weak boxes.

Run extensions/build_weak_labels.py first. Metrics are measured against the weak boxes, so they
show agreement with the measurement code and not with human annotation. A box gives the rough
position of a crack. It says nothing about width or structural severity.

    python extensions/localization_yolo.py --epochs 30
    python extensions/localization_yolo.py --eval_only --checkpoint ext_work/models/crack_yolo_best.pt
"""
import argparse
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extensions.shared import RESULTS_DIR, WORK_DIR, save_json  # noqa: E402


def draw(im, boxes, color):
    for (x0, y0, x1, y1) in boxes:
        cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), color, 1)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(WORK_DIR / "weak"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "localization"))
    ap.add_argument("--work", default=str(WORK_DIR))
    ap.add_argument("--model", default="yolo11n.pt", help="starting weights (falls back to yolov8n.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="0", help="GPU index, or 'cpu'")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--checkpoint", default=None, help="weights to evaluate with --eval_only")
    args = ap.parse_args()

    from ultralytics import YOLO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    os.environ["WANDB_DISABLED"] = "true"
    device = int(args.device) if args.device.isdigit() else args.device
    data = os.path.abspath(args.data)
    yaml_path = os.path.join(data, "data.yaml")
    yaml.safe_dump({"path": data, "train": "images/train", "val": "images/val", "test": "images/test",
                    "names": {0: "crack"}}, open(yaml_path, "w"))

    if args.eval_only:
        best = args.checkpoint or os.path.join(args.work, "models", "crack_yolo_best.pt")
        run_dir = None
    else:
        try:
            det = YOLO(args.model)
        except Exception:
            det = YOLO("yolov8n.pt")
        det.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                  device=device, project=os.path.join(args.work, "yolo_runs"), name="crack_det", exist_ok=True,
                  patience=10, plots=True, verbose=False)
        run_dir = str(det.trainer.save_dir)
        best = os.path.join(run_dir, "weights", "best.pt")
        os.makedirs(os.path.join(args.work, "models"), exist_ok=True)
        shutil.copy(best, os.path.join(args.work, "models", "crack_yolo_best.pt"))
    print("Evaluating", best)

    det = YOLO(best)
    m = det.val(data=yaml_path, split="test", imgsz=args.imgsz, batch=args.batch, device=device, verbose=False)
    box = {"precision": float(m.box.mp), "recall": float(m.box.mr),
           "map50": float(m.box.map50), "map50_95": float(m.box.map)}
    print("\nBox metrics on the test split (against weak boxes)")
    print({k: round(v, 3) for k, v in box.items()})

    man = pd.read_csv(os.path.join(data, "manifest.csv"))
    te = man[man.split == "test"].reset_index(drop=True)
    files = [os.path.join(data, "images", "test", [x for x in os.listdir(os.path.join(data, "images", "test"))
                                                     if x.startswith(s)][0]) for s in te.stem]
    has_box = [len(r.boxes) > 0 for r in det.predict(source=files, conf=args.conf, imgsz=args.imgsz,
                                                      stream=True, verbose=False)]
    p, r_, f, _ = precision_recall_fscore_support(te.is_crack.astype(int).values, np.array(has_box).astype(int),
                                                  average="binary", zero_division=0)
    img_level = {"conf_threshold": args.conf, "precision": float(p), "recall": float(r_), "f1": float(f)}
    print(f"Image level, crack present (conf >= {args.conf}): precision {p:.3f}  recall {r_:.3f}  F1 {f:.3f}")

    os.makedirs(args.out, exist_ok=True)
    sel = te[te.is_crack].sample(min(8, int(te.is_crack.sum())), random_state=0)
    fig, axes = plt.subplots(2, len(sel), figsize=(2.8 * len(sel), 6), squeeze=False)
    for j, s in enumerate(sel.stem):
        fp = os.path.join(data, "images", "test",
                          [x for x in os.listdir(os.path.join(data, "images", "test")) if x.startswith(s)][0])
        im = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]
        gt = []
        for line in open(os.path.join(data, "labels", "test", s + ".txt")):
            _, cx, cy, bw, bh = map(float, line.split())
            gt.append(((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
        res = det.predict(source=fp, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        pb = res.boxes.xyxy.cpu().numpy().tolist()
        axes[0, j].imshow(draw(im.copy(), gt, (0, 255, 0))); axes[0, j].set_title("weak box (green)", fontsize=8)
        axes[1, j].imshow(draw(im.copy(), pb, (255, 0, 0))); axes[1, j].set_title(f"predicted ({len(pb)})", fontsize=8)
        axes[0, j].axis("off"); axes[1, j].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "localization_examples.png"), dpi=150); plt.close(fig)

    if run_dir and os.path.exists(os.path.join(run_dir, "results.csv")):
        shutil.copy(os.path.join(run_dir, "results.csv"), os.path.join(args.out, "training_log.csv"))
    save_json({"labels": "weak boxes from classical crack measurement", "weights": os.path.basename(best),
               "epochs": None if args.eval_only else args.epochs, "imgsz": args.imgsz,
               "test_box_metrics": box, "test_image_level": img_level},
              os.path.join(args.out, "localization_results.json"))


if __name__ == "__main__":
    main()
