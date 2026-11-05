# Self-Supervised Learning vs. Supervised Pretraining for Bridge Crack Detection

This repository accompanies the paper *"Exploring Self-Supervised Learning for Bridge Crack Classification in Structural Health Monitoring "* and reproduces the
label-efficiency comparison between four ImageNet-pretrained backbones on the
[SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) bridge-crack dataset:

| Method                | Backbone     | Params |
|-----------------------|--------------|--------|
| Supervised (DeiT)     | ViT-S/16     | 22 M   |
| DINOv2                | ViT-S/14     | 22 M   |
| MoCo v3               | ViT-S/16     | 22 M   |
| MAE                   | ViT-B/16     | 86 M   |

For each method we report linear-probe and full fine-tune accuracy at 5 / 10 / 20 / 50 / 100 %
of the labeled training data.

---

## What is original and what is new

The files at the top level (the `evaluate_*.py` scripts, `check_dataset.py`, `create_better_plots.py`,
`requirements.txt`, `setup_env.sh` and the `results/` folder) come from the original study and are unchanged.
The additions live in `severity/`, `extensions/`, `notebooks/`, `results_extension/` and `paper/`.
The only existing files edited are this README and `.gitignore`.

| Task | Original study | This extension |
|------|----------------|----------------|
| Crack / no crack | Linear probe and fine-tuning, four backbones | Reused as stage 1 (DINOv2 linear probe) |
| Severity (Minor / Moderate / Severe) | Not covered | Proxy labels from measured crack extent, then a head on frozen DINOv2 features |
| Crack localization | Not covered | YOLO trained on weak boxes |
| Crack segmentation | Not covered | U-Net trained on weak masks |

The original code is at https://github.com/gkrisp98/SSL-For-Bridge-Crack-Classification. If you use this repository, cite the original paper named above and the SDNET2018 dataset, as well as this extension (see section 8).

---

## 1. Environment setup

Tested with Python 3.10 + PyTorch 2.x + CUDA 11/12.

```bash
pip install -r requirements.txt
```

(Or use `bash setup_env.sh` for the interactive HPC/conda flow.)

The MAE and MoCo v3 weights are downloaded automatically on first run via `timm` and
the official Facebook research URLs. DINOv2 is loaded from `torch.hub`.

## 2. Dataset preparation

Download SDNET2018 and arrange it as follows:

```
archive/
├── Decks/
│   ├── CD/        # Cracked deck images
│   └── UD/        # Uncracked deck images
├── Pavements/
│   ├── CP/
│   └── UP/
└── Walls/
    ├── CW/
    └── UW/
```

You can verify the structure with:

```bash
python check_dataset.py --data_dir /path/to/archive
```

## 3. Reproducing the results

The four-model table is built incrementally — `results/results.json` is the artifact
that all plots and the LaTeX table read from.

```bash
# Step 1: base evaluation (DINOv2, MAE, plus other reference models)
python evaluate_fair_comparison.py \
    --data_dir /path/to/archive \
    --output_dir results \
    --percentages 0.05 0.1 0.2 0.5 1.0 \
    --skip_ijepa

# Step 2: add Supervised ViT-S/16 (and MoCo v2)
python create_better_plots.py --data_dir /path/to/archive

# Step 3: add MoCo v3 ViT-S/16
python evaluate_moco_v3.py --data_dir /path/to/archive
```

## 4. File overview

| File | Purpose |
|------|---------|
| `evaluate_fair_comparison.py` | Main pipeline — DINOv2, MAE, DINO, supervised ResNet50, optional I-JEPA |
| `create_better_plots.py`      | Adds Supervised ViT-S/16 and MoCo v2; emits plots + LaTeX |
| `evaluate_moco_v3.py`         | Adds MoCo v3 ViT-S/16 to `results.json` |
| `evaluate_mae_small.py`       | Standalone MAE evaluation utility |
| `plot.py`                     | Final publication figure from `results.json` |
| `check_dataset.py`            | Sanity-checks the SDNET2018 directory layout |
| `requirements.txt`            | Python dependencies |
| `setup_env.sh`                | Optional conda/pip setup helper |
| `results/`                    | Pre-computed results (`results.json`, figures, LaTeX table) |

## 5. Notes on reproducibility

* All models are ImageNet-pretrained.
* Train/test split: stratified 80/20 with `random_state=42`.
* Linear probe and fine-tune use the same nested-subset sampling so the 5 % subset
  is contained in the 10 % subset, which is contained in the 20 % subset, etc.
* Fine-tuning runs for 50 epochs by default; lower-resource GPUs can use
  `--eval_epochs 30` with a small accuracy hit.

## 6. Extension: crack severity (Minor / Moderate / Severe)

The original study stops at crack / no crack. The `severity/` folder adds a second stage. Images flagged as
cracked get a severity level predicted from the same frozen DINOv2 ViT-S/14 features.

```
image -> DINOv2 (frozen) -> crack / no crack -> (crack) -> severity head -> Minor | Moderate | Severe
```

SDNET2018 has no severity annotations, so the labels here are proxy labels. They are computed from
measurements of the crack in the image (length, width, coverage) and then cut into three classes. They are
not engineering ground truth, and a photograph alone cannot show whether a bridge is safe. Treat the output
as a screening aid for a qualified inspector to review. No results for this extension are stored in the
repository. You produce them by running the scripts below.

### 6.1 Setup

```bash
pip install -r requirements.txt              # original dependencies, plus PyTorch and torchvision
pip install -r requirements-extensions.txt   # scikit-image, joblib, ultralytics and a few others
python severity/selftest.py                  # optional: checks the crack measurement on synthetic images
```

### 6.2 Run order

```bash
# 1. Measure cracks and create proxy severity labels (CPU, measurements are cached)
python severity/make_severity_labels.py --data_dir /path/to/archive --spotcheck 12

# 2. Extract frozen DINOv2 features for every image, once (GPU)
python severity/extract_features.py --data_dir /path/to/archive

# 3. Stage 1: fit and save the crack / no crack linear probe
python severity/train_crack_detector.py

# 4. Stage 2: train and evaluate the severity head at 5 / 10 / 20 / 50 / 100 % of the labels
python severity/train_severity.py

# 5. Run both stages on new images
python severity/infer_two_stage.py path/to/image_or_folder --csv predictions.csv
```

Results go to `results/severity/` and models to `severity/models/` (git-ignored).

### 6.3 How the proxy labels are made

1. Each cracked image is measured with classical image processing and no extra model. The steps are
   illumination flattening, a multi-scale ridge filter for dark thin lines, a hysteresis threshold,
   skeletonisation, then length (skeleton pixels), width (half-depth rule along the centreline) and
   crack coverage. The thresholds adapt to image noise, so plain texture is not read as a crack.
2. One score (`--score area|width|length`, default `area`) is split into three classes. With
   `--bins quantile` (the default) the cut-offs are terciles fitted on the training split only. Use
   `--quantiles 0.5 0.85` for a Minor-heavy distribution. With `--bins fixed --thresholds T1 T2` you
   supply your own cut-offs. Pass `--mm_per_px` only if you know the real scale of your images.
3. Cracked images where nothing could be measured are excluded. `--keep_undetected` labels them Minor.
4. `--spotcheck N` writes a contact sheet and a CSV template. Fill in `human_severity` (0, 1 or 2) and run
   `python severity/make_severity_labels.py --score_spotcheck severity/spotcheck_template.csv` to get the
   agreement and quadratic-weighted kappa between the automatic labels and a person.

### 6.4 Evaluation in `severity/train_severity.py`

The script evaluates on the same stratified 80/20 split as the original experiments. It reports accuracy,
macro precision, recall and F1, per-class F1, the confusion matrix, the Minor-to-Severe confusion rate and
quadratic-weighted kappa. Smaller label fractions use nested, class-stratified subsets averaged over
`--seeds`. Class imbalance is handled with a class-balanced loss (`--no_class_weight` turns it off).

Two reference points are included. One is the majority class. The other is a classifier on the raw crack
measurements. That one is circular, because the labels come from those measurements, so read it as a rough
ceiling and as a check on how much of the DINOv2 score is the labeling rule being learned back.

When a stage 1 model exists, the script also reports how many true cracks reach stage 2 and the severity
accuracy when stage 1 misses count as errors.

### 6.5 Files

| File | Purpose |
|------|---------|
| `severity/measure_cracks.py` | Classical crack measurement (length, width, coverage) |
| `severity/make_severity_labels.py` | Proxy labels, cut-offs, spot-check tools |
| `severity/extract_features.py` | Cache frozen DINOv2 features for all images |
| `severity/train_crack_detector.py` | Stage 1: save the crack / no crack linear probe |
| `severity/train_severity.py` | Stage 2: severity head, label-efficiency sweep, metrics, plots |
| `severity/infer_two_stage.py` | Run both stages on images or folders |
| `severity/common.py` | Dataset listing and the split used by the original scripts |
| `severity/selftest.py` | Synthetic-image test of the measurement code |

### 6.6 Limitations

* The proxy labels depend on a heuristic crack detector. Shadows, joints, stains and rough texture can produce
  false ridges, and very faint cracks can be missed. Look at the spot-check sheet before training.
* Widths are measured on 256x256 patches and are relative unless you calibrate with `--mm_per_px`.
* Quantile cut-offs are relative to this dataset and do not come from any structural standard.
* Inference expects SDNET2018-style close-up patches. Tile larger photos first.
* Softmax confidence values are not calibrated probabilities.

## 7. Further extensions: validated severity, localization, segmentation

These scripts live in `extensions/`. They split the data by source photo (the number before the first `-`
in a file name, plus the surface type), so patches from one photo never appear in both training and test
data. The original experiments split patches at random, so numbers from `extensions/` are not directly
comparable with `severity/train_severity.py`.

```bash
# Severity with a validation split, tuned regularisation, a Severe-as-Minor count and a human-review table
python extensions/severity_validated.py
# Optional: fine-tune the last two DINOv2 blocks and compare with the frozen head on the same test images
python extensions/severity_finetune.py --epochs 5 --unfreeze 2

# Weak boxes and masks for localization and segmentation (look at results_extension/weak_labels_preview.png)
python extensions/build_weak_labels.py --archive /path/to/archive

python extensions/localization_yolo.py --epochs 30
python extensions/segmentation_unet.py --epochs 15

# Post-hoc analyses reported in the paper: stage 1 thresholds and recall breakdowns, other heads, severity per surface,
# calibration, ROC, score regression, and the two-stage pipeline outcome (needs the saved models in severity/models/)
python extensions/post_hoc_analysis.py

# LaTeX tables for the paper, built from the result files
python extensions/make_paper_tables.py
```

Weak labels need a clear caveat. SDNET2018 has no bounding boxes or pixel masks, so `build_weak_labels.py`
generates them with the same measurement code that makes the severity labels. A model trained on them learns
to reproduce that code. Its scores measure agreement with the weak labels and not accuracy against human
annotation. For a fair evaluation, hand-annotate a small test set (30 to 50 images is a reasonable start),
put the files into `ext_work/weak/masks/test` and `ext_work/weak/labels/test`, delete any `*.cache` files in
`ext_work/weak/labels`, and rerun the evaluation.

Small result files (JSON, PNG, CSV) are written to `results_extension/`. Large files (the weak dataset,
model weights, YOLO run folders) go to `ext_work/`, which is git-ignored.

`notebooks/Bridge_Crack_Full_Pipeline_Colab_executed.ipynb` is the notebook, with its outputs, from the run that produced
the numbers in the paper. The scripts above are converted from its cells. A shorter Colab runner that calls the
scripts is in `notebooks/MINI_PROJECT_FINAL_CODE.ipynb`. It expects the
project zip and the dataset zip in a Google Drive folder named `MINI_PROJECT`, or a repository URL set in
its third setup step.

