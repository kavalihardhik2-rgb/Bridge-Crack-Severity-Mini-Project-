#!/usr/bin/env python3
"""
Stage 1 (existing task): crack / no-crack linear probe on frozen DINOv2 features.

Same recipe as the original repo's linear probe (StandardScaler + LogisticRegression, C=0.1,
stratified 80/20 split with seed 42), but the fitted model is SAVED so the two-stage
pipeline can use it.

Phase-1 sanity check: results/results.json reports ~94.2% for DINOv2 ViT-S/14 at 100% labels.
Our number can differ slightly because features here are extracted without training-time
augmentation, but it should be in the same ballpark.

  python severity/train_crack_detector.py
"""
import argparse
import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from severity.common import CRACK_LABEL, repo_split  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(HERE, "features_dinov2_vits14.npz"))
    ap.add_argument("--output", default=os.path.join(HERE, "models", "crack_detector.joblib"))
    ap.add_argument("--C", type=float, default=0.1)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=False)
    X, y = d["features"], d["targets"]
    train_idx, test_idx = repo_split(y)

    scaler = StandardScaler().fit(X[train_idx])
    clf = LogisticRegression(max_iter=1000, C=args.C, solver="lbfgs", random_state=42)
    clf.fit(scaler.transform(X[train_idx]), y[train_idx])
    pred = clf.predict(scaler.transform(X[test_idx]))

    acc = accuracy_score(y[test_idx], pred)
    p, r, f, _ = precision_recall_fscore_support(
        y[test_idx], pred, labels=[CRACK_LABEL], zero_division=0)
    print(f"Stage 1 (crack vs no crack) test accuracy: {acc:.4f}")
    print(f"  'Crack' class  precision={p[0]:.4f}  recall={r[0]:.4f}  F1={f[0]:.4f}")
    print("  (compare with results/results.json -> linear -> 'DINOv2 ViT-S/14' -> '1.0')")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump({"scaler": scaler, "clf": clf, "crack_label": CRACK_LABEL,
                 "test_accuracy": float(acc)}, args.output)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
