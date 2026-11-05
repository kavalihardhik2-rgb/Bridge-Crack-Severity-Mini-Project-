#!/usr/bin/env python3
"""
Post-hoc analyses reported in the paper. Everything here is computed from saved artifacts
(DINOv2 features, severity labels, and the saved stage 1 and severity models), so no GPU is needed.

Sections
  A. Dataset and label statistics by surface, and measurement statistics by proxy class
  B. Stage 1 (original split): thresholds, recall by class and surface and by crack extent,
     ROC AUC and average precision, other heads
  C. Severity, original split: surface-only baseline, errors versus distance to the cut-offs
  D. Severity, photo-aware split: refit of the frozen head, per-surface results, calibration,
     ROC AUC, other heads, regression on the log score
  E. Pipeline: outcome table and the effect of the stage 1 threshold (original split)

    python extensions/post_hoc_analysis.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extensions.shared import (DEFAULT_FEATURES, DEFAULT_LABELS, REPO_ROOT, SEV_NAMES, assign_split,  # noqa: E402
                               save_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(DEFAULT_FEATURES))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--cutoffs", default=str(REPO_ROOT / "severity" / "severity_cutoffs.json"))
    ap.add_argument("--detector", default=str(REPO_ROOT / "severity" / "models" / "crack_detector.joblib"))
    ap.add_argument("--severity_head", default=str(REPO_ROOT / "severity" / "models" / "severity_head.joblib"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results_extension" / "severity" / "post_hoc_analysis.json"))
    args = ap.parse_args()

    import joblib
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score, confusion_matrix,
                                 f1_score, precision_recall_curve, roc_auc_score)
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    d = np.load(args.features, allow_pickle=False)
    X_all, targets, surfaces = d["features"], d["targets"], d["surfaces"]
    lab_all = pd.read_csv(args.labels)
    lab = lab_all[lab_all.severity >= 0].reset_index(drop=True)
    cuts = np.array(json.load(open(args.cutoffs))["cutoffs"], dtype=float)
    out = {}

    # ------------------------------------------------------------------ A
    out["A_composition"] = {s: {"images": int((surfaces == s).sum()), "cracked": int(((surfaces == s) & (targets == 0)).sum())}
                            for s in ["Decks", "Pavements", "Walls"]}
    g = lab_all.groupby("surface")
    out["A_labels_by_surface"] = {s: {"cracked": int(len(x)), "no_crack_found": int((x.severity < 0).sum()),
                                      **{n: int((x.severity == k).sum()) for k, n in enumerate(SEV_NAMES)}} for s, x in g}
    out["A_measurements_by_class"] = {
        n: {c: [float(lab[lab.severity == k][c].quantile(q)) for q in (0.25, 0.5, 0.75)]
            for c in ("length_px", "p90_width_px", "coverage")} for k, n in enumerate(SEV_NAMES)}
    out["A_spearman"] = {f"{a}_vs_{b}": float(spearmanr(lab[a], lab[b])[0]) for a, b in
                         [("score", "length_px"), ("score", "p90_width_px"), ("score", "n_components"), ("length_px", "p90_width_px")]}

    # ------------------------------------------------------------------ B  stage 1, original split
    tr_all, te_all = train_test_split(list(range(len(targets))), test_size=0.2, stratify=targets, random_state=42)
    te1, tr1 = np.array(te_all), np.array(tr_all)
    det = joblib.load(args.detector)
    p1 = det["clf"].predict_proba(det["scaler"].transform(X_all[te1]))[:, list(det["clf"].classes_).index(0)]
    y1 = (targets[te1] == 0).astype(int)
    thr_rows = []
    for th in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
        pred = p1 >= th
        tp, fp, fn = int((pred & (y1 == 1)).sum()), int((pred & (y1 == 0)).sum()), int((~pred & (y1 == 1)).sum())
        P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        thr_rows.append({"threshold": th, "TP": tp, "FP": fp, "FN": fn, "precision": P, "recall": R,
                         "f1": 2 * P * R / max(P + R, 1e-9), "accuracy": float(((pred == (y1 == 1)).mean()))})
    pr, rc, _ = precision_recall_curve(y1, p1)
    out["B_stage1"] = {"test_images": int(len(te1)), "cracked": int(y1.sum()), "thresholds": thr_rows,
                       "roc_auc": float(roc_auc_score(y1, p1)), "average_precision": float(average_precision_score(y1, p1)),
                       "recall_at_precision_0.8": float(rc[pr >= 0.8].max())}
    sub = lab_all[lab_all.idx.isin(set(te1.tolist()))].copy()
    pmap = dict(zip(te1, p1))
    sub["p"] = [pmap[i] for i in sub.idx]
    rec = {}
    for name, m in [("Minor", sub.severity == 0), ("Moderate", sub.severity == 1), ("Severe", sub.severity == 2),
                    ("no_measurable_crack", sub.severity < 0)]:
        rec[name] = {f"thr_{th}": float((sub[m].p >= th).mean()) for th in (0.5, 0.3, 0.1)} | {"n": int(m.sum())}
    for s in ["Decks", "Pavements", "Walls"]:
        m = sub.surface == s
        rec[s] = {"thr_0.5": float((sub[m].p >= 0.5).mean()), "n": int(m.sum())}
    out["B_stage1_recall_by_group"] = rec
    sl = sub[sub.severity >= 0].copy()
    sl["bin"] = pd.qcut(sl.score, 8, labels=False)
    out["B_stage1_recall_by_extent_octile"] = [{"bin": int(k), "n": int(len(x)), "median_score": float(x.score.median()),
                                                 "recall": float((x.p >= 0.5).mean())} for k, x in sl.groupby("bin")]
    sc1 = StandardScaler().fit(X_all[tr1])
    Xtr1, Xte1, ytr1 = sc1.transform(X_all[tr1]), sc1.transform(X_all[te1]), (targets[tr1] == 0).astype(int)

    def head_row(name, score, pred=None):
        pred = (score >= 0.5) if pred is None else pred
        tp, fp, fn = int((pred & (y1 == 1)).sum()), int((pred & (y1 == 0)).sum()), int((~pred & (y1 == 1)).sum())
        P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        pr_, rc_, _ = precision_recall_curve(y1, score)
        return {"head": name, "accuracy": float((pred == (y1 == 1)).mean()), "precision": P, "recall": R,
                "f1": 2 * P * R / max(P + R, 1e-9), "recall_at_precision_0.8": float(rc_[pr_ >= 0.8].max())}
    heads = []
    heads.append(head_row("LR C=0.1 (refit)", LogisticRegression(C=0.1, max_iter=1000).fit(Xtr1, ytr1).predict_proba(Xte1)[:, 1]))
    heads.append(head_row("LR class-balanced", LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced").fit(Xtr1, ytr1).predict_proba(Xte1)[:, 1]))
    heads.append(head_row("LR C=1", LogisticRegression(C=1.0, max_iter=2000).fit(Xtr1, ytr1).predict_proba(Xte1)[:, 1]))
    sv = LinearSVC(C=0.01, class_weight="balanced", max_iter=20000).fit(Xtr1, ytr1).decision_function(Xte1)
    heads.append(head_row("Linear SVM class-balanced", sv, pred=sv >= 0))
    heads.append(head_row("MLP 256", MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True, random_state=0,
                                                   max_iter=200).fit(Xtr1, ytr1).predict_proba(Xte1)[:, 1]))
    out["B_stage1_other_heads"] = heads

    # ------------------------------------------------------------------ C  severity, original split
    X = X_all[lab.idx.values]
    y = lab.severity.values.astype(int)
    tr_o = lab[lab.idx.isin(set(tr_all))].index.values
    te_o = lab[lab.idx.isin(set(te_all))].index.values
    sc = StandardScaler().fit(X[tr_o])
    clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr_o]), y[tr_o])
    pred, yt = clf.predict(sc.transform(X[te_o])), y[te_o]
    surf = lab.surface.values
    maj = {s: int(np.bincount(y[tr_o][surf[tr_o] == s], minlength=3).argmax()) for s in np.unique(surf)}
    dist = np.min(np.abs(lab.score.values[te_o][:, None] - cuts[None, :]) / cuts[None, :], axis=1)
    rows = []
    for a, b in zip([0, 0.1, 0.25, 0.5, 1], [0.1, 0.25, 0.5, 1, 10]):
        m = (dist >= a) & (dist < b)
        rows.append({"relative_distance": f"{a}-{b}", "n": int(m.sum()), "accuracy": float((pred[m] == yt[m]).mean())})
    err = pred != yt
    out["C_severity_original"] = {
        "accuracy": float((pred == yt).mean()), "surface_only_baseline": float(np.mean([yt[i] == maj[surf[te_o][i]] for i in range(len(te_o))])),
        "accuracy_by_surface": {s: {"n": int((surf[te_o] == s).sum()), "accuracy": float((pred[surf[te_o] == s] == yt[surf[te_o] == s]).mean())}
                                for s in np.unique(surf)},
        "accuracy_by_distance": rows, "errors": int(err.sum()), "errors_one_class_away": int((np.abs(pred - yt)[err] == 1).sum())}

    # ------------------------------------------------------------------ D  severity, photo-aware split
    sp = lab.path.map(assign_split).values
    tr, va, te = [np.where(sp == s)[0] for s in ("train", "val", "test")]
    scp = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = [scp.transform(X[i]) for i in (tr, va, te)]
    best_c, best_f = None, -1
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        f = f1_score(y[va], LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, y[tr]).predict(Xva), average="macro")
        if f > best_f:
            best_c, best_f = C, f
    head = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(Xtr, y[tr])
    proba = head.predict_proba(Xte)
    pa, yp = proba.argmax(1), y[te]
    cm = confusion_matrix(yp, pa, labels=[0, 1, 2])
    conf = proba.max(1)
    cal, ece = [], 0.0
    for a, b in [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]:
        m = (conf >= a) & (conf < b)
        if m.sum():
            cal.append({"bin": f"{a}-{min(b, 1)}", "n": int(m.sum()), "mean_confidence": float(conf[m].mean()),
                        "accuracy": float((pa[m] == yp[m]).mean())})
            ece += m.sum() / len(yp) * abs(conf[m].mean() - (pa[m] == yp[m]).mean())
    sfe = lab.surface.values[te]

    def alt(name, clf_):
        clf_.fit(Xtr, y[tr])
        p_ = clf_.predict(Xte)
        c_ = confusion_matrix(yp, p_, labels=[0, 1, 2])
        return {"head": name, "val_macro_f1": float(f1_score(y[va], clf_.predict(Xva), average="macro")),
                "test_accuracy": float(accuracy_score(yp, p_)), "test_macro_f1": float(f1_score(yp, p_, average="macro")),
                "severe_as_minor": int(c_[2, 0])}
    alts = [alt("LR used", LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced")),
            alt("LR no class weights", LogisticRegression(C=best_c, max_iter=3000))]
    Xr = [X[i] for i in (tr, va, te)]
    lr_ns = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(Xr[0], y[tr])
    c_ns = confusion_matrix(yp, lr_ns.predict(Xr[2]), labels=[0, 1, 2])
    alts.append({"head": "LR no standardization", "val_macro_f1": float(f1_score(y[va], lr_ns.predict(Xr[1]), average="macro")),
                 "test_accuracy": float(accuracy_score(yp, lr_ns.predict(Xr[2]))),
                 "test_macro_f1": float(f1_score(yp, lr_ns.predict(Xr[2]), average="macro")), "severe_as_minor": int(c_ns[2, 0])})
    best_svm = max([alt(f"Linear SVM C={C}", LinearSVC(C=C, class_weight="balanced", max_iter=20000)) for C in (0.0003, 0.001, 0.003, 0.01, 0.03)],
                   key=lambda r: r["val_macro_f1"])
    best_knn = max([alt(f"kNN k={k}", KNeighborsClassifier(n_neighbors=k, weights="distance")) for k in (5, 15, 31, 61)],
                   key=lambda r: r["val_macro_f1"])
    alts += [best_svm, best_knn, alt("MLP 256", MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True,
                                                               random_state=0, max_iter=300))]
    # regression on the log score, then the same cut-offs
    z = np.log10(lab.score.values)
    best_r = None
    for a in [10, 100, 300, 1000, 3000, 10000, 30000]:
        r = Ridge(alpha=a).fit(Xtr, z[tr])
        mae = float(np.abs(r.predict(Xva) - z[va]).mean())
        if best_r is None or mae < best_r[0]:
            best_r = (mae, a, r)
    pz = best_r[2].predict(Xte)
    preg = np.digitize(pz, np.log10(cuts))
    cr = confusion_matrix(yp, preg, labels=[0, 1, 2])
    out["D_severity_photo_aware"] = {
        "sizes": [int(len(tr)), int(len(va)), int(len(te))], "chosen_C": best_c, "accuracy": float(accuracy_score(yp, pa)),
        "macro_f1": float(f1_score(yp, pa, average="macro")), "quadratic_kappa": float(cohen_kappa_score(yp, pa, weights="quadratic")),
        "confusion_matrix": cm.tolist(), "within_one_class": float(np.mean(np.abs(pa - yp) <= 1)),
        "by_surface": {s: {"n": int((sfe == s).sum()), "accuracy": float((pa[sfe == s] == yp[sfe == s]).mean()),
                           "severe_as_minor": int(((yp[sfe == s] == 2) & (pa[sfe == s] == 0)).sum())} for s in ["Decks", "Pavements", "Walls"]},
        "calibration": cal, "expected_calibration_error": float(ece),
        "ovr_roc_auc": {n: float(roc_auc_score((yp == k).astype(int), proba[:, k])) for k, n in enumerate(SEV_NAMES)},
        "other_heads": alts,
        "regression": {"alpha": best_r[1], "mae_log10": float(np.abs(pz - z[te]).mean()), "spearman": float(spearmanr(pz, z[te])[0]),
                       "accuracy": float(accuracy_score(yp, preg)), "macro_f1": float(f1_score(yp, preg, average="macro")),
                       "quadratic_kappa": float(cohen_kappa_score(yp, preg, weights="quadratic")),
                       "severe_as_minor": int(cr[2, 0]), "within_one_class": float(np.mean(np.abs(preg - yp) <= 1)),
                       "confusion_matrix": cr.tolist()}}

    # ------------------------------------------------------------------ E  pipeline (original split, saved severity head)
    sev = joblib.load(args.severity_head)
    sp_all = sev["clf"].predict(sev["scaler"].transform(X_all[te1]))
    lab_i = lab_all.set_index("idx")
    ytrue = np.array([lab_i.loc[g_, "severity"] if g_ in lab_i.index else -9 for g_ in te1])
    sl_m, unc = ytrue >= 0, targets[te1] == 1
    outcome = {}
    for k, n in enumerate(SEV_NAMES):
        m = ytrue == k
        fl = p1 >= 0.5
        outcome[n] = {"n": int(m.sum()), **{f"graded_{SEV_NAMES[j]}": int((m & fl & (sp_all == j)).sum()) for j in range(3)},
                      "not_flagged": int((m & ~fl).sum())}
    thr_pipe = []
    for th in [0.5, 0.3, 0.1]:
        fl = p1 >= th
        thr_pipe.append({"threshold": th, "images_flagged": int(fl.sum()), "cracks_passed": int((fl & sl_m).sum()),
                         "end_to_end_accuracy": float(((sp_all == ytrue) & fl & sl_m).sum() / sl_m.sum()),
                         "severe_missed_or_graded_minor": int(((ytrue == 2) & (~fl | (sp_all == 0))).sum()),
                         "uncracked_flagged": int((fl & unc).sum()), "uncracked_graded_severe": int((fl & unc & (sp_all == 2)).sum())})
    out["E_pipeline"] = {"outcome_at_0.5": outcome, "threshold_effect": thr_pipe}

    # ------------------------------------------------------------------ report
    print("Stage 1 @0.5:", {k: round(v, 3) for k, v in thr_rows[0].items()})
    print("Stage 1 ROC AUC %.3f  AP %.3f  recall@P>=0.8 %.3f" % (out["B_stage1"]["roc_auc"], out["B_stage1"]["average_precision"],
                                                                 out["B_stage1"]["recall_at_precision_0.8"]))
    print("Photo-aware frozen head: acc %.4f macro-F1 %.4f kappa %.3f within-one %.3f ECE %.3f"
          % (accuracy_score(yp, pa), f1_score(yp, pa, average="macro"), cohen_kappa_score(yp, pa, weights="quadratic"),
             np.mean(np.abs(pa - yp) <= 1), ece))
    print("Regression: macro-F1 %.4f kappa %.3f" % (out["D_severity_photo_aware"]["regression"]["macro_f1"],
                                                    out["D_severity_photo_aware"]["regression"]["quadratic_kappa"]))
    print("Pipeline @0.5: e2e accuracy %.3f, flagged %d" % (thr_pipe[0]["end_to_end_accuracy"], thr_pipe[0]["images_flagged"]))
    save_json(out, args.out)


if __name__ == "__main__":
    main()
