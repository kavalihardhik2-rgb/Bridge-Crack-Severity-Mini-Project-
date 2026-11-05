#!/usr/bin/env python3
"""
Derive PROXY severity labels (Minor / Moderate / Severe) for SDNET2018 crack images
from measurable crack properties (Option 3 of the planned enhancement).

Steps
  1. list the dataset exactly like the original repo and reproduce its 80/20 split
  2. measure every *cracked* image (class 0) with classical image processing
     (cached to severity/crack_measurements.csv, so this runs once)
  3. turn one score (area / width / length) into 3 classes
        --bins quantile : cut-offs = quantiles of the TRAIN split only (no test leakage)
        --bins fixed    : user-supplied thresholds (e.g. in mm when --mm_per_px is known)
  4. write severity/severity_labels.csv + severity/severity_cutoffs.json
  5. optional human spot-check tooling (contact sheet, CSV template, agreement score)

These labels are heuristics, NOT engineering ground truth. See README for caveats.

Examples
  python severity/make_severity_labels.py --data_dir /path/to/archive
  python severity/make_severity_labels.py --data_dir /path/to/archive --score width --bins quantile --quantiles 0.5 0.85
  python severity/make_severity_labels.py --data_dir /path/to/archive --score width --bins fixed \\
         --mm_per_px 0.25 --thresholds 0.3 1.0           # only if you know the real scale!
  python severity/make_severity_labels.py --data_dir /path/to/archive --spotcheck 12
  python severity/make_severity_labels.py --score_spotcheck severity/spotcheck_template.csv
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from severity.common import (CRACK_LABEL, SEVERITY_NAMES, list_dataset, repo_split)  # noqa: E402
from severity.measure_cracks import (METRIC_KEYS, MeasureConfig, measure_crack,  # noqa: E402
                                     measure_crack_file, overlay)

try:
    from tqdm import tqdm
except ImportError:  # tqdm is in requirements.txt, but don't hard-fail without it
    def tqdm(x, **kw):
        return x

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- measuring
def _worker(path):
    return measure_crack_file(path, MeasureConfig())


def measure_all(paths, workers):
    if workers <= 1:
        return [_worker(p) for p in tqdm(paths, desc="Measuring cracks")]
    with Pool(workers) as pool:
        return list(tqdm(pool.imap(_worker, paths, chunksize=32), total=len(paths),
                         desc="Measuring cracks"))


def build_measurements(args):
    cache = os.path.join(HERE, "crack_measurements.csv")
    if os.path.exists(cache) and not args.remeasure:
        print(f"Re-using cached measurements: {cache}  (use --remeasure to recompute)")
        return pd.read_csv(cache)

    paths, targets, surfaces = list_dataset(args.data_dir)
    train_idx, test_idx = repo_split(targets)
    split = np.array(["train"] * len(paths), dtype=object)
    split[test_idx] = "test"

    crack_idx = np.where(targets == CRACK_LABEL)[0]
    print(f"Dataset: {len(paths)} images, {len(crack_idx)} cracked -> measuring cracked images")
    results = measure_all([paths[i] for i in crack_idx], args.workers)

    df = pd.DataFrame(results)
    df.insert(0, "idx", crack_idx)          # index into the global, repo-ordered image list
    df.insert(1, "path", [paths[i] for i in crack_idx])
    df.insert(2, "surface", surfaces[crack_idx])
    df.insert(3, "split", split[crack_idx])
    df.to_csv(cache, index=False)
    print(f"Saved measurements -> {cache}")
    return df


# --------------------------------------------------------------------------- binning
def compute_score(df, score, mm_per_px):
    """Return (score values, unit string)."""
    if score == "area":
        return df["coverage"].values, "fraction of image"
    if score == "width":
        col = df["p90_width_px"].values
        return (col * mm_per_px, "mm") if mm_per_px else (col, "px")
    if score == "length":
        col = df["length_px"].values
        return (col * mm_per_px, "mm") if mm_per_px else (col, "px")
    raise ValueError(score)


def assign_classes(values, detected, train_mask, args):
    """Map score -> {0,1,2}; undetected cracks get -1."""
    if args.bins == "fixed":
        if args.thresholds is None:
            sys.exit("--bins fixed requires --thresholds T1 T2")
        cuts = sorted(args.thresholds)
    else:
        q = sorted(args.quantiles)
        fit = values[train_mask & detected]          # cut-offs from TRAIN crack images only
        cuts = [float(np.quantile(fit, qi)) for qi in q]
    labels = np.full(len(values), -1, dtype=int)
    labels[detected] = np.digitize(values[detected], cuts)   # 0,1,2 for two cut-offs
    return labels, [float(c) for c in cuts]


# --------------------------------------------------------------------------- spot-check
def _load_overlay(path, size=192):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    m, mask, skel = measure_crack(img, MeasureConfig(), return_masks=True)
    a = cv2.resize(img, (size, size))
    b = cv2.resize(overlay(img, mask, skel), (size, size), interpolation=cv2.INTER_NEAREST)
    return np.vstack([a, b])


def make_spotcheck(df, n, seed=0):
    rng = np.random.RandomState(seed)
    lab = df[df["severity"] >= 0]
    cols, template_rows = [], []
    for c, name in enumerate(SEVERITY_NAMES):
        sub = lab[lab["severity"] == c]
        pick = sub.iloc[rng.permutation(len(sub))[:n]]
        tiles = []
        for _, r in pick.iterrows():
            tiles.append(_load_overlay(r["path"]))
            template_rows.append({"path": r["path"], "auto_severity": int(r["severity"]),
                                  "human_severity": ""})
        if tiles:
            col = np.hstack(tiles)
            cv2.putText(col, f"AUTO: {name}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
            cols.append(col)
    if cols:
        width = max(c.shape[1] for c in cols)
        cols = [np.pad(c, ((0, 0), (0, width - c.shape[1]), (0, 0))) for c in cols]
        sheet = np.vstack(cols)
        out_png = os.path.join(HERE, "spotcheck_contact_sheet.png")
        cv2.imwrite(out_png, sheet)
        print(f"Contact sheet (top = image, bottom = detected crack; one row per auto class): {out_png}")
    tmpl = os.path.join(HERE, "spotcheck_template.csv")
    pd.DataFrame(template_rows).sample(frac=1, random_state=seed).to_csv(tmpl, index=False)
    print(f"Spot-check template: {tmpl}\n"
          f"  -> fill 'human_severity' with 0=Minor 1=Moderate 2=Severe (judge the image, ideally blind\n"
          f"     to auto_severity / ideally by someone with civil-engineering experience), then run:\n"
          f"     python severity/make_severity_labels.py --score_spotcheck {tmpl}")


def score_spotcheck(path):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
    d = pd.read_csv(path).dropna(subset=["human_severity"])
    if d.empty:
        sys.exit("No human_severity values filled in yet.")
    a, h = d["auto_severity"].astype(int), d["human_severity"].astype(int)
    print(f"Rated images: {len(d)}")
    print(f"Agreement (accuracy): {accuracy_score(h, a):.3f}")
    print(f"Quadratic-weighted Cohen's kappa: {cohen_kappa_score(h, a, weights='quadratic'):.3f}")
    print("Confusion matrix (rows = human, cols = auto):")
    print(confusion_matrix(h, a, labels=[0, 1, 2]))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", help="SDNET2018 'archive' directory (Decks/Pavements/Walls)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--remeasure", action="store_true", help="ignore cached measurements")
    ap.add_argument("--score", choices=["area", "width", "length"], default="area",
                    help="area=cleaned crack coverage (length x width), width=90th-pct width, length=skeleton length")
    ap.add_argument("--bins", choices=["quantile", "fixed"], default="quantile")
    ap.add_argument("--quantiles", type=float, nargs=2, default=[1 / 3, 2 / 3],
                    help="quantile cut-offs for Minor|Moderate and Moderate|Severe (default: terciles)")
    ap.add_argument("--thresholds", type=float, nargs=2, default=None,
                    help="fixed cut-offs in the score's unit (for --bins fixed)")
    ap.add_argument("--mm_per_px", type=float, default=None,
                    help="OPTIONAL physical scale. Only set this if you know it for your images.")
    ap.add_argument("--keep_undetected", action="store_true",
                    help="label cracked images where nothing was measured as Minor instead of excluding them")
    ap.add_argument("--spotcheck", type=int, default=0, metavar="N",
                    help="write a contact sheet + CSV template with N images per class for manual review")
    ap.add_argument("--score_spotcheck", metavar="CSV", help="score a filled-in spot-check CSV and exit")
    args = ap.parse_args()

    if args.score_spotcheck:
        score_spotcheck(args.score_spotcheck)
        return
    labels_csv = os.path.join(HERE, "severity_labels.csv")

    if args.spotcheck and not args.data_dir and os.path.exists(labels_csv):
        make_spotcheck(pd.read_csv(labels_csv), args.spotcheck)
        return
    if not args.data_dir and not os.path.exists(os.path.join(HERE, "crack_measurements.csv")):
        sys.exit("--data_dir is required the first time (to measure cracks).")

    df = build_measurements(args)
    detected = ((df["length_px"] > 0) & (df.get("read_error", 0) == 0)).values
    if args.keep_undetected:
        detected = np.ones(len(df), dtype=bool)
    values, unit = compute_score(df, args.score, args.mm_per_px)
    train_mask = (df["split"] == "train").values

    labels, cuts = assign_classes(values, detected, train_mask, args)
    df["score"], df["severity"] = values, labels
    df["severity_name"] = [SEVERITY_NAMES[l] if l >= 0 else "undetected" for l in labels]
    df.to_csv(labels_csv, index=False)

    info = {"score": args.score, "unit": unit, "bins": args.bins, "cutoffs": cuts,
            "quantiles": sorted(args.quantiles) if args.bins == "quantile" else None,
            "mm_per_px": args.mm_per_px,
            "note": "PROXY labels derived from image measurements; not engineering ground truth."}
    with open(os.path.join(HERE, "severity_cutoffs.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("\n=== Severity label summary ===")
    print(f"score = {args.score} [{unit}], cut-offs = {[round(c, 5) for c in cuts]}")
    print(f"cracked images measured : {len(df)}")
    print(f"undetected (excluded)   : {int((labels == -1).sum())}")
    for c, n in enumerate(SEVERITY_NAMES):
        for sp in ("train", "test"):
            k = int(((labels == c) & (df['split'] == sp).values).sum())
            print(f"  {n:<9}{sp:<6}{k:>6}")
    print(f"Saved -> {labels_csv}")
    if args.spotcheck:
        make_spotcheck(df, args.spotcheck)


if __name__ == "__main__":
    main()
