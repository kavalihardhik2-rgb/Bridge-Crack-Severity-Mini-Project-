#!/usr/bin/env python3
"""
Feature 1 (A and B): severity head with a proper validation split.

* Splits by source photo into train / validation / test (see shared.py).
* Picks the regularisation strength C on the validation set only, then evaluates once on test.
* Reports the dangerous error "Severe predicted as Minor" explicitly.
* Shows how many predictions would go to a human reviewer at several confidence thresholds.

Severity labels are proxy labels (measured crack extent), not engineering ground truth.

    python extensions/severity_validated.py
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extensions.shared import (DEFAULT_FEATURES, DEFAULT_LABELS, REPO_ROOT, RESULTS_DIR, SEV_NAMES,  # noqa: E402
                               fit_head, load_severity_split, report_dict, save_json)


def plot_confusion(cm, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(SEV_NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(SEV_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (proxy label)"); ax.set_title("Severity, test set")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(DEFAULT_FEATURES))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--out", default=str(RESULTS_DIR / "severity"))
    ap.add_argument("--model_out", default=str(REPO_ROOT / "severity" / "models" / "severity_head_tuned.joblib"))
    ap.add_argument("--review_threshold", type=float, default=0.6)
    args = ap.parse_args()

    from sklearn.metrics import f1_score
    data = load_severity_split(args.features, args.labels)
    X, y, tr, va, te = data["X"], data["y"], data["tr"], data["va"], data["te"]
    print(f"train={len(tr)}  val={len(va)}  test={len(te)}")
    print("class counts, train:", np.bincount(y[tr], minlength=3), " test:", np.bincount(y[te], minlength=3))

    best_f1, best_c, val_scores = -1.0, None, {}
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        sc, clf = fit_head(X[tr], y[tr], C)
        f1 = f1_score(y[va], clf.predict(sc.transform(X[va])), labels=[0, 1, 2], average="macro", zero_division=0)
        val_scores[str(C)] = float(f1)
        print(f"C={C:<6} validation macro-F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_c = f1, C
    print("Chosen C (validation):", best_c)

    sc, clf = fit_head(X[tr], y[tr], best_c)       # the test set is used only from here on
    proba = clf.predict_proba(sc.transform(X[te]))
    pred, yt = clf.classes_[proba.argmax(1)], y[te]
    rep = report_dict(yt, pred)

    print("\nTest results, frozen DINOv2 + linear head")
    for n, v in rep["per_class"].items():
        print(f"  {n:<9} precision {v['precision']:.3f}  recall {v['recall']:.3f}  F1 {v['f1']:.3f}  n={v['support']}")
    print(f"  macro-F1 {rep['macro_f1']:.4f}   accuracy {rep['accuracy']:.4f}")
    ce = rep["critical_errors"]
    print(f"  Severe predicted as Minor: {ce['severe_as_minor']} of {ce['severe_total']}")
    print(f"  Severe predicted as Moderate: {ce['severe_as_moderate']} of {ce['severe_total']}")

    conf = proba.max(1)
    review = []
    for t in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        keep = conf >= t
        n = int(keep.sum())
        review.append({
            "min_confidence": t,
            "auto_accepted_pct": round(100 * n / len(conf), 1),
            "sent_to_human_pct": round(100 * (1 - n / len(conf)), 1),
            "accuracy_on_accepted": round(float((pred[keep] == yt[keep]).mean()), 3) if n else None,
            "severe_as_minor_still_accepted": int(((yt == 2) & (pred == 0) & keep).sum()),
        })
    print("\nHuman-review table")
    print(pd.DataFrame(review).to_string(index=False))
    print(f"\nPredictions below {args.review_threshold} would be marked for human review. "
          "Low confidence means a person should look. It does not mean the crack is safe or dangerous.")

    os.makedirs(args.out, exist_ok=True)
    plot_confusion(rep["confusion_matrix"], os.path.join(args.out, "severity_confusion_matrix.png"))
    save_json({"split": "by source photo, about 70/15/15", "labels": "proxy labels from measured crack extent",
               "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "chosen_C": best_c, "validation_macro_f1_by_C": val_scores,
               "test": rep, "review_table": review, "review_threshold": args.review_threshold},
              os.path.join(args.out, "severity_validated_results.json"))
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump({"scaler": sc, "clf": clf, "classes": SEV_NAMES, "C": best_c}, args.model_out)
    print("Saved model ->", args.model_out)


if __name__ == "__main__":
    main()
