#!/usr/bin/env python3
"""
Stage 2 (new): train + evaluate a severity head (Minor / Moderate / Severe) on frozen DINOv2 features.

What this script does
  * Label-efficiency sweep: trains on nested, class-stratified subsets (5/10/20/50/100 % of the
    severity-labeled TRAIN images; several seeds for the small subsets) and always evaluates on the
    full held-out TEST split (same 80/20 split as the original repo).
  * Metrics: accuracy, macro precision / recall / F1, per-class F1, confusion matrix,
    Minor<->Severe ("two-level") error rate, quadratic-weighted kappa.
  * Reference baselines: majority class, and a classifier on the raw crack measurements.
    The latter is CIRCULAR by construction (the labels are derived from those measurements) and is
    included only so you can see how much of DINOv2's score is "re-learning the labeling heuristic".
  * End-to-end two-stage numbers (if a stage-1 detector is available): what fraction of severity
    ground-truth cracks reach stage 2, and severity accuracy when stage-1 misses count as errors.
  * Saves the 100 % model for inference, JSON results and plots.

  python severity/train_severity.py
  python severity/train_severity.py --percentages 0.05 0.1 0.2 0.5 1.0 --seeds 5
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from severity.common import SEVERITY_NAMES, nested_subset  # noqa: E402
from severity.measure_cracks import METRIC_KEYS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABELS = [0, 1, 2]


# ------------------------------------------------------------------ metrics
def evaluate(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS,
                                                 average="macro", zero_division=0)
    _, _, f_each, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS,
                                                      zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    two_level = float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) == 2))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
        "f1_per_class": {n: float(v) for n, v in zip(SEVERITY_NAMES, f_each)},
        "minor_severe_confusion_rate": two_level,
        "quadratic_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "confusion_matrix": cm.tolist(),
    }


def fit_head(Xtr, ytr, C, balanced=True):
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=C, solver="lbfgs", random_state=42,
                             class_weight="balanced" if balanced else None)
    clf.fit(scaler.transform(Xtr), ytr)
    return scaler, clf


def mean_std(runs, key):
    v = [r[key] for r in runs]
    return float(np.mean(v)), float(np.std(v))


# ------------------------------------------------------------------ plots
def plot_confusion(cm, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    cm_norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(SEVERITY_NAMES); ax.set_yticklabels(SEVERITY_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (proxy label)"); ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


def plot_label_efficiency(sweep, baselines, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pcts = sorted(float(k) for k in sweep)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, name in zip(axes, ["macro_f1", "accuracy"], ["Macro F1", "Accuracy"]):
        m = [sweep[str(p)][key + "_mean"] for p in pcts]
        s = [sweep[str(p)][key + "_std"] for p in pcts]
        ax.errorbar([p * 100 for p in pcts], m, yerr=s, marker="o", capsize=3, label="DINOv2 + linear head")
        if key == "accuracy":
            ax.axhline(baselines["majority_accuracy"], ls=":", c="gray", label="majority class")
        ax.set_xscale("log"); ax.set_xticks([p * 100 for p in pcts])
        ax.set_xticklabels([f"{p*100:g}" for p in pcts])
        ax.set_xlabel("% of severity-labeled training data"); ax.set_ylabel(name)
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Severity classification label efficiency (proxy labels)")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(HERE, "features_dinov2_vits14.npz"))
    ap.add_argument("--labels", default=os.path.join(HERE, "severity_labels.csv"))
    ap.add_argument("--cutoffs", default=os.path.join(HERE, "severity_cutoffs.json"))
    ap.add_argument("--crack_detector", default=os.path.join(HERE, "models", "crack_detector.joblib"))
    ap.add_argument("--output_dir", default=os.path.join(ROOT, "results", "severity"))
    ap.add_argument("--model_out", default=os.path.join(HERE, "models", "severity_head.joblib"))
    ap.add_argument("--percentages", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, default=3, help="random subsets averaged for fractions < 1.0")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--no_class_weight", action="store_true",
                    help="disable class-balanced loss (on by default to handle class imbalance)")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    d = np.load(args.features, allow_pickle=False)
    X_all, paths_all = d["features"], d["paths"]
    lab = pd.read_csv(args.labels)
    lab = lab[lab["severity"] >= 0].reset_index(drop=True)

    # alignment check: label rows must point at the same image in the feature file
    assert (paths_all[lab["idx"].values] == lab["path"].values).all(), \
        "severity_labels.csv and the feature file were built from different dataset listings."

    gidx = lab["idx"].values
    y = lab["severity"].values.astype(int)
    is_train = (lab["split"] == "train").values
    tr_pos, te_pos = np.where(is_train)[0], np.where(~is_train)[0]
    X = X_all[gidx]
    print(f"Severity-labeled cracks: train={len(tr_pos)}  test={len(te_pos)}")
    for c, n in enumerate(SEVERITY_NAMES):
        print(f"  {n:<9} train={int((y[tr_pos]==c).sum()):>5}  test={int((y[te_pos]==c).sum()):>5}")

    # ---- baselines
    maj = int(np.bincount(y[tr_pos]).argmax())
    baselines = {"majority_class": SEVERITY_NAMES[maj],
                 "majority_accuracy": float(np.mean(y[te_pos] == maj))}
    meas_path = os.path.join(HERE, "crack_measurements.csv")
    if os.path.exists(meas_path):
        meas = pd.read_csv(meas_path).set_index("idx").loc[gidx]
        M = meas[METRIC_KEYS].values.astype(float)
        sc, clf = fit_head(M[tr_pos], y[tr_pos], C=10.0, balanced=not args.no_class_weight)
        baselines["raw_measurement_classifier"] = evaluate(y[te_pos], clf.predict(sc.transform(M[te_pos])))
        baselines["raw_measurement_classifier"]["note"] = (
            "CIRCULAR reference: labels are derived from these measurements, so this is roughly an upper bound.")
        print(f"\nReference (circular) raw-measurement classifier: acc="
              f"{baselines['raw_measurement_classifier']['accuracy']:.3f}  macro-F1="
              f"{baselines['raw_measurement_classifier']['macro_f1']:.3f}")
    print(f"Majority-class baseline accuracy: {baselines['majority_accuracy']:.3f}")

    # ---- label-efficiency sweep
    sweep, final = {}, None
    print("\n%labels  n_train   accuracy        macro-F1        Minor<->Severe")
    for frac in args.percentages:
        runs = []
        n_seeds = 1 if frac >= 1.0 else args.seeds
        for s in range(n_seeds):
            sub = nested_subset(tr_pos, y[tr_pos], frac, seed=42 + s)
            scaler, clf = fit_head(X[sub], y[sub], args.C, balanced=not args.no_class_weight)
            res = evaluate(y[te_pos], clf.predict(scaler.transform(X[te_pos])))
            res["n_train"] = int(len(sub))
            runs.append(res)
            if frac >= 1.0:
                final = (scaler, clf, res)
        entry = {"n_train": runs[0]["n_train"], "n_seeds": n_seeds}
        for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1",
                  "minor_severe_confusion_rate", "quadratic_kappa"]:
            entry[k + "_mean"], entry[k + "_std"] = mean_std(runs, k)
        entry["confusion_matrix_first_seed"] = runs[0]["confusion_matrix"]
        entry["f1_per_class_first_seed"] = runs[0]["f1_per_class"]
        sweep[str(frac)] = entry
        print(f"{frac*100:>6g}  {entry['n_train']:>7}   {entry['accuracy_mean']:.3f}±{entry['accuracy_std']:.3f}"
              f"    {entry['macro_f1_mean']:.3f}±{entry['macro_f1_std']:.3f}"
              f"    {entry['minor_severe_confusion_rate_mean']:.3f}")

    results = {"baselines": baselines, "label_efficiency": sweep,
               "class_weight_balanced": not args.no_class_weight,
               "labels_are": "proxy labels derived from image measurements (see severity_cutoffs.json)"}

    # ---- final model, confusion matrix, saved head
    if final is not None:
        scaler, clf, res = final
        results["final_100pct"] = res
        plot_confusion(res["confusion_matrix"], os.path.join(args.output_dir, "severity_confusion_matrix.png"),
                       "Severity (DINOv2 linear head, 100% labels)")
        cutoffs = json.load(open(args.cutoffs)) if os.path.exists(args.cutoffs) else None
        os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
        joblib.dump({"scaler": scaler, "clf": clf, "classes": SEVERITY_NAMES, "label_cutoffs": cutoffs},
                    args.model_out)
        print(f"\nSaved severity head -> {args.model_out}")
        print("Confusion matrix (rows = true, cols = predicted; Minor/Moderate/Severe):")
        print(np.array(res["confusion_matrix"]))
    plot_label_efficiency(sweep, baselines, os.path.join(args.output_dir, "severity_label_efficiency.png"))

    # ---- end-to-end two-stage evaluation
    if final is not None and os.path.exists(args.crack_detector):
        det = joblib.load(args.crack_detector)
        s1_pred = det["clf"].predict(det["scaler"].transform(X[te_pos]))
        reached = s1_pred == det["crack_label"]
        s2_pred = final[1].predict(final[0].transform(X[te_pos]))
        end_to_end_correct = reached & (s2_pred == y[te_pos])
        results["two_stage_end_to_end"] = {
            "severity_labeled_test_cracks": int(len(te_pos)),
            "passed_by_stage1": int(reached.sum()),
            "stage1_recall_on_these": float(reached.mean()),
            "severity_accuracy_given_detected": float(np.mean(s2_pred[reached] == y[te_pos][reached])) if reached.any() else None,
            "severity_accuracy_end_to_end_misses_count_as_wrong": float(end_to_end_correct.mean()),
        }
        e = results["two_stage_end_to_end"]
        print(f"\nTwo-stage: stage 1 passes {e['passed_by_stage1']}/{e['severity_labeled_test_cracks']} true cracks "
              f"(recall {e['stage1_recall_on_these']:.3f}); end-to-end severity accuracy "
              f"{e['severity_accuracy_end_to_end_misses_count_as_wrong']:.3f}")
    elif final is not None:
        print("\n(No stage-1 detector found - run train_crack_detector.py for end-to-end numbers.)")

    out_json = os.path.join(args.output_dir, "severity_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results -> {out_json}")


if __name__ == "__main__":
    main()
