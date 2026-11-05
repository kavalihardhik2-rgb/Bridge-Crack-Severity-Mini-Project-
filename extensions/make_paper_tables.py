#!/usr/bin/env python3
"""
Turn the JSON files in results_extension/ into LaTeX tables in paper/tables/.
Every number in the paper tables then comes from a result file, not from copy and paste.
Tables whose result file does not exist yet are skipped.

    python extensions/make_paper_tables.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extensions.shared import REPO_ROOT, RESULTS_DIR, SEV_NAMES  # noqa: E402

OUT = REPO_ROOT / "paper" / "tables"


def load(rel):
    p = RESULTS_DIR / rel
    return json.load(open(p)) if p.exists() else None


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(OUT / name, "w") as f:
        f.write(body)
    print("wrote", OUT / name)


def tabular(header, rows, spec):
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def severity_table():
    r = load("severity/severity_validated_results.json")
    if not r:
        return
    t = r["test"]
    rows = [[n, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", str(v["support"])]
            for n, v in t["per_class"].items()]
    rows.append(["Macro average", "", "", f"{t['macro_f1']:.3f}", str(sum(v["support"] for v in t["per_class"].values()))])
    write("severity_test.tex", tabular(["Class", "Precision", "Recall", "F1", "Images"], rows, "lrrrr"))
    ce = t["critical_errors"]
    write("severity_critical.tex", tabular(
        ["Error", "Count", "Of severe images"],
        [["Severe predicted as Minor", str(ce["severe_as_minor"]), str(ce["severe_total"])],
         ["Severe predicted as Moderate", str(ce["severe_as_moderate"]), str(ce["severe_total"])]], "lrr"))
    ft = load("severity/severity_finetuned_results.json")
    if ft:
        write("severity_frozen_vs_finetuned.tex", tabular(
            ["Model", "Macro-F1", "Severe as Minor"],
            [["Frozen encoder", f"{ft['frozen_reference']['macro_f1']:.3f}",
              str(ft["frozen_reference"]["critical_errors"]["severe_as_minor"])],
             [f"Fine-tuned ({ft['unfrozen_blocks']} blocks)", f"{ft['test']['macro_f1']:.3f}",
              str(ft["test"]["critical_errors"]["severe_as_minor"])]], "lrr"))


def localization_table():
    r = load("localization/localization_results.json")
    if not r:
        return
    b, im = r["test_box_metrics"], r["test_image_level"]
    write("localization_test.tex", tabular(
        ["Metric", "Value"],
        [["Box precision", f"{b['precision']:.3f}"], ["Box recall", f"{b['recall']:.3f}"],
         ["mAP@0.5", f"{b['map50']:.3f}"], ["mAP@0.5:0.95", f"{b['map50_95']:.3f}"],
         [f"Image-level F1 (conf {im['conf_threshold']})", f"{im['f1']:.3f}"]], "lr"))


def segmentation_table():
    r = load("segmentation/segmentation_results.json")
    if not r:
        return
    t = r["test"]
    write("segmentation_test.tex", tabular(
        ["Metric", "Value"],
        [["IoU", f"{t['IoU']:.3f}"], ["Dice", f"{t['Dice']:.3f}"], ["Precision", f"{t['precision']:.3f}"],
         ["Recall", f"{t['recall']:.3f}"], ["Threshold (chosen on validation)", f"{r['threshold']}"]], "lr"))


if __name__ == "__main__":
    severity_table(); localization_table(); segmentation_table()
