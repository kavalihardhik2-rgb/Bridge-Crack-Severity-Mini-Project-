#!/usr/bin/env python3
"""
Two-stage inference:  image -> DINOv2 features -> crack / no crack -> (if crack) severity.

  python severity/infer_two_stage.py path/to/img.jpg other.jpg
  python severity/infer_two_stage.py path/to/folder --csv predictions.csv

Output per image, e.g.
  bridge_crack.jpg
    Crack detected : YES  (confidence 0.97)
    Severity       : MODERATE  (confidence 0.61)
    Inspection priority (indicative only): MEDIUM

IMPORTANT
  * This is an AI-assisted SCREENING aid, not a structural assessment. Severity classes are learned
    from PROXY labels (measured crack extent in images), not from engineering criteria, and a photo
    cannot establish structural safety. All outputs need review by a qualified inspector.
  * The models were trained on SDNET2018-style 256x256 patches at close range. Whole-bridge photos
    should be cropped/tiled into comparable patches first.
  * 'confidence' is the classifier's softmax probability, which is NOT a calibrated probability.
"""
import argparse
import csv
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from severity.common import IMG_EXTENSIONS, get_transform, load_dinov2_vits14  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PRIORITY = {"Minor": "LOW", "Moderate": "MEDIUM", "Severe": "HIGH"}


def collect_images(inputs):
    out = []
    for p in inputs:
        if os.path.isdir(p):
            for dp, _, fns in sorted(os.walk(p)):
                out += [os.path.join(dp, f) for f in sorted(fns) if f.lower().endswith(IMG_EXTENSIONS)]
        elif os.path.isfile(p):
            out.append(p)
        else:
            print(f"Skipping (not found): {p}", file=sys.stderr)
    return out


def main():
    import torch
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("--crack_detector", default=os.path.join(HERE, "models", "crack_detector.joblib"))
    ap.add_argument("--severity_head", default=os.path.join(HERE, "models", "severity_head.joblib"))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--csv", help="optional path to write predictions as CSV")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    for f in (args.crack_detector, args.severity_head):
        if not os.path.exists(f):
            sys.exit(f"Missing model file: {f}\nRun train_crack_detector.py and train_severity.py first.")
    det, sev = joblib.load(args.crack_detector), joblib.load(args.severity_head)
    images = collect_images(args.inputs)
    if not images:
        sys.exit("No images found.")

    model, _ = load_dinov2_vits14(args.device)
    tf = get_transform()
    rows = []
    for start in range(0, len(images), args.batch_size):
        chunk = images[start:start + args.batch_size]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(args.device)
        with torch.no_grad():
            feats = model(x).float().cpu().numpy()

        p1 = det["clf"].predict_proba(det["scaler"].transform(feats))
        crack_col = list(det["clf"].classes_).index(det["crack_label"])
        p_crack = p1[:, crack_col]
        p2 = sev["clf"].predict_proba(sev["scaler"].transform(feats))

        for i, path in enumerate(chunk):
            is_crack = p_crack[i] >= 0.5
            row = {"image": path, "crack": "YES" if is_crack else "NO",
                   "crack_confidence": round(float(p_crack[i] if is_crack else 1 - p_crack[i]), 3),
                   "severity": "", "severity_confidence": "", "inspection_priority": ""}
            if is_crack:
                k = int(np.argmax(p2[i]))
                name = sev["classes"][int(sev["clf"].classes_[k])]   # column -> class label -> name
                row.update(severity=name.upper(), severity_confidence=round(float(p2[i][k]), 3),
                           inspection_priority=PRIORITY[name])
            rows.append(row)

            print(os.path.basename(path))
            print(f"  Crack detected : {row['crack']}  (confidence {row['crack_confidence']})")
            if is_crack:
                print(f"  Severity       : {row['severity']}  (confidence {row['severity_confidence']})")
                print(f"  Inspection priority (indicative only): {row['inspection_priority']}")

    print("\nNote: screening aid only - severity is learned from proxy labels and must be confirmed "
          "by a qualified inspector.")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
