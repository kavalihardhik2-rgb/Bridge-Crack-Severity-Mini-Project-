# Bridge crack severity, localization and segmentation

This repository extends a self-supervised bridge crack classifier. The original study by Gkrispanis, Nousias and Borrmann
tells you whether an image contains a crack. This project adds three things on top of it: how severe the crack is, where it
is, and which pixels it covers.

The classifier, the evaluation scripts at the top level and the SDNET2018 baseline experiments come from the original
authors. Their code is at https://github.com/gkrisp98/SSL-For-Bridge-Crack-Classification. The dataset is SDNET2018,
described at https://digitalcommons.usu.edu/all_datasets/48/. Everything in `severity/`, `extensions/`, `notebooks/`,
`results_extension/` and `paper/` is the extension.

## The problem with the dataset

SDNET2018 only says crack or no crack. It has no severity ratings, no bounding boxes and no pixel masks. So the labels in
this project are generated: a classical image-processing procedure measures each crack and turns the measurements into a
severity class, a mask and a set of boxes. They are proxy and weak labels, not expert annotation. Every score for severity,
boxes and masks below measures agreement with that procedure. It does not show accuracy against human judgment.

## What was built

| Task | Original study | This extension |
|------|----------------|----------------|
| Crack or no crack | Linear probe and fine-tuning on four encoders | Reused as stage 1 (frozen DINOv2 ViT-S/14, linear probe) |
| Severity (Minor, Moderate, Severe) | Not covered | Proxy labels from measured crack extent, then a head on the frozen features |
| Crack localization | Not covered | YOLO11n trained on weak boxes |
| Crack segmentation | Not covered | U-Net trained on weak masks |

## Results at a glance

| Component | Result | Scored against |
|-----------|--------|----------------|
| Stage 1, crack detection | 94.16% accuracy, crack recall 0.693 | Dataset labels |
| Severity, frozen head | Macro-F1 0.703 | Proxy labels |
| Severity, last two blocks fine-tuned | Macro-F1 0.768 | Proxy labels |
| Localization, YOLO11n | mAP50 0.643 | Weak boxes |
| Segmentation, U-Net | Dice 0.783, IoU 0.643 | Weak masks |
| Both stages together | End-to-end severity accuracy 52.2% | Proxy labels |

The stage 1 result matches the 94.2% of the original study, but only about 15% of the images are cracked and the classifier
finds roughly 69% of them. Missed cracks never reach the severity stage, so that recall limits the whole pipeline. The full
tables, the threshold study and the limitations are in the paper.

## Repository layout

| Folder or file | What it holds |
|----------------|---------------|
| `evaluate_*.py`, `create_better_plots.py`, `check_dataset.py`, `requirements.txt`, `setup_env.sh`, `results/` | The original study's scripts and results, unchanged |
| `severity/` | Crack measurement, proxy labels, feature extraction, the stage 1 and severity heads, two-stage inference |
| `extensions/` | Validated severity, fine-tuning, weak labels, YOLO11n, U-Net, post-hoc analyses, paper tables |
| `notebooks/` | The executed Colab notebook that produced the paper's numbers, and a shorter runner that calls the scripts |
| `results_extension/` | Small result files written by the extension scripts |
| `paper/` | A LaTeX skeleton and the folder where generated tables go |

## Setup

```bash
pip install -r requirements.txt
pip install -r requirements-extensions.txt
```

Install PyTorch and torchvision for your CUDA version first if you have not already. Download SDNET2018 and arrange it as
`Decks`, `Pavements` and `Walls` folders, each holding a cracked and an uncracked subfolder (`CD`/`UD`, `CP`/`UP`, `CW`/`UW`).
The dataset itself is not in this repository.

## Running it

The severity pipeline, in order:

```bash
python severity/selftest.py                                   # optional check of the crack measurement
python severity/make_severity_labels.py --data_dir /path/to/archive --spotcheck 12
python severity/extract_features.py --data_dir /path/to/archive    # needs a GPU, run once
python severity/train_crack_detector.py
python severity/train_severity.py
python severity/infer_two_stage.py path/to/image_or_folder
```

Open `severity/spotcheck_contact_sheet.png` after the first step. If the traced cracks do not follow real cracks, stop and
fix that before training anything.

The extensions, which split the data by source photograph:

```bash
python extensions/severity_validated.py
python extensions/severity_finetune.py            # optional
python extensions/build_weak_labels.py --archive /path/to/archive
python extensions/localization_yolo.py
python extensions/segmentation_unet.py
python extensions/post_hoc_analysis.py
python extensions/make_paper_tables.py
```

Large files (features, models, the weak-label dataset, YOLO runs) go to `severity/models/` and `ext_work/`, which git
ignores. The numbers in the paper come from `notebooks/Bridge_Crack_Full_Pipeline_Colab_executed.ipynb`. The scripts are
converted from its cells and have not been run end to end as the source of those numbers.

## Splits

The original study splits patches at random (stratified 80/20, seed 42). Patches cut from one photograph look alike, so the
extensions also use a split by source photograph, taking the number before the hyphen in a file name as the photograph. The
two splits give numbers that cannot be compared directly, and the paper says which one each result uses.

## Limitations

The proxy labels come from a heuristic measurement, with no expert review. SDNET2018 has no pixel size, so crack size is
relative and nothing is in millimetres. The procedure finds no measurable crack in about 20% of the cracked images, and in
46% of the cracked pavement images, and those are left out of the severity labels. Fine-tuning, YOLO11n and the U-Net come
from single runs. A photograph cannot show whether a bridge is safe, so treat every output as a screening aid that a
qualified inspector must check.

## Licence and citation

This repository has no licence file yet. The original repository it builds on has none either, so the reuse terms of the
original code are for its authors to set. Check the SDNET2018 terms before sharing data, and note that Ultralytics YOLO is
distributed under AGPL-3.0.

If you use this code, please cite the original study and the dataset, and link to this repository:

* K. Gkrispanis, S. Nousias and A. Borrmann, "Exploring self-supervised learning for bridge crack classification in structural health monitoring."
* S. Dorafshan, R. J. Thomas and M. Maguire, "SDNET2018: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks," Data in Brief, vol. 21, pp. 1664-1668, 2018, doi: 10.1016/j.dib.2018.11.015.
