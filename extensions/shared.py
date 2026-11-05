"""
Helpers shared by the scripts in extensions/.

Splitting: SDNET2018 patches are cut from larger photos, and patches from one photo look alike.
The original experiments split patches at random. The extensions split by source photo instead
(the number before the first "-" in the file name, plus the surface type), so a photo never ends up
in both training and test data. Because of that, numbers from these scripts are not directly
comparable with severity/train_severity.py, which uses the original random split.

This module needs only numpy, pandas and scikit-learn (no torch).
"""
import json
import os
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES = REPO_ROOT / "severity" / "features_dinov2_vits14.npz"
DEFAULT_LABELS = REPO_ROOT / "severity" / "severity_labels.csv"
WORK_DIR = REPO_ROOT / "ext_work"            # large, regenerable files (git-ignored)
RESULTS_DIR = REPO_ROOT / "results_extension"  # small result files that are committed

SEV_NAMES = ["Minor", "Moderate", "Severe"]


def group_key(path):
    """Surface + photo id, e.g. 'Decks:7001' for .../Decks/CD/7001-12.jpg."""
    p = str(path).replace("\\", "/").split("/")
    surface = p[-3] if len(p) >= 3 else ""
    return f"{surface}:{re.split(r'[-_.]', p[-1])[0]}"


def assign_split(path, test_pct=15, val_pct=15):
    """Deterministic split by photo: about 70% train, 15% val, 15% test."""
    h = zlib.crc32(group_key(path).encode()) % 100
    return "test" if h < test_pct else ("val" if h < test_pct + val_pct else "train")


def load_labels(labels_csv=DEFAULT_LABELS):
    if not os.path.exists(labels_csv):
        raise SystemExit(f"Missing {labels_csv}. Run severity/make_severity_labels.py first.")
    return pd.read_csv(labels_csv)


def load_severity_split(features=DEFAULT_FEATURES, labels_csv=DEFAULT_LABELS):
    """Features, severity labels and train/val/test indices for the severity experiments."""
    if not os.path.exists(features):
        raise SystemExit(f"Missing {features}. Run severity/extract_features.py first.")
    d = np.load(features, allow_pickle=False)
    lab = load_labels(labels_csv)
    lab = lab[lab.severity >= 0].reset_index(drop=True)
    if not (d["paths"][lab.idx.values] == lab.path.values).all():
        raise SystemExit("The feature file and the label file were built from different dataset listings.")
    X = d["features"][lab.idx.values]
    y = lab.severity.values.astype(int)
    sp = lab.path.map(assign_split).values
    tr, va, te = [np.where(sp == s)[0] for s in ("train", "val", "test")]
    return {"lab": lab, "X": X, "y": y, "tr": tr, "va": va, "te": te}


def fit_head(Xtr, ytr, C):
    """Standardised, class-balanced logistic regression (the frozen-encoder head)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def report_dict(y_true, y_pred):
    """Per-class precision/recall/F1, macro-F1, accuracy and confusion matrix as plain python types."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    sev_total = int(cm[2].sum())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "per_class": {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, n in enumerate(SEV_NAMES)},
        "confusion_matrix": cm.tolist(),
        "critical_errors": {"severe_as_minor": int(cm[2, 0]), "severe_as_moderate": int(cm[2, 1]),
                            "severe_total": sev_total},
    }


def save_json(obj, path):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
    print("Saved", path)
