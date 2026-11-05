"""
Classical (no-learning) crack measurement used to derive *proxy* severity labels.

Pipeline for one image:
  1. grayscale + illumination flattening (subtract a heavily blurred copy)
  2. multi-scale ridge filter for dark, thin structures (Sato filter, black ridges)
  3. hysteresis threshold (absolute + noise-adaptive floor) -> binary crack mask
  4. remove small blobs, skeletonize, drop short skeleton fragments (texture noise)
  5. re-measure width from the intensity profile (half-depth rule), then measure:
        length_px        number of skeleton pixels (approximate crack length)
        mean_width_px    mean of 2 * distance-transform of the half-depth mask along the skeleton
        p90_width_px     90th percentile of the same (robust "wide part" measure)
        max_width_px
        area_px          pixels of the cleaned crack mask
        coverage         area_px / image area
        n_components     number of surviving crack fragments

IMPORTANT LIMITATIONS
  * These are heuristic measurements on 256x256 patches, not engineering-grade
    crack-width readings. Widths are biased by the ridge-filter scale and are only
    meaningful *relative to each other* unless you calibrate with --mm_per_px.
  * Concrete texture, joints, stains and shadows can produce false ridges; the
    spot-check tooling in make_severity_labels.py exists to catch this.
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sato, threshold_otsu, apply_hysteresis_threshold
from skimage.morphology import skeletonize, remove_small_objects

METRIC_KEYS = [
    "length_px", "mean_width_px", "p90_width_px", "max_width_px",
    "area_px", "coverage", "n_components",
]


@dataclass
class MeasureConfig:
    flatten_sigma: float = 25.0          # background blur for illumination flattening
    sato_sigmas: tuple = (1.0, 1.5, 2.0, 3.0)
    ridge_floor: float = 4.0             # absolute floor on the ridge response (rejects faint texture)
    ridge_median_mult: float = 10.0      # adaptive floor: this x median ridge response (scales with image noise)
    width_depth_frac: float = 0.5        # width = extent where darkness exceeds this fraction of local crack depth
    width_search_radius: int = 10        # max distance (px) from the centreline searched when measuring width
    low_ratio: float = 0.5               # hysteresis low = low_ratio * high
    min_area_px: int = 20                # remove blobs smaller than this
    min_skeleton_px: int = 15            # remove skeleton fragments shorter than this


def _remove_small(mask, min_px):
    """Version-safe wrapper (scikit-image renamed min_size -> max_size in 0.26)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return remove_small_objects(mask, min_size=min_px)


def _empty_metrics():
    return {k: 0.0 for k in METRIC_KEYS}


def measure_crack(img_bgr: np.ndarray, cfg: MeasureConfig = None, return_masks: bool = False):
    """Return a dict of crack metrics (and optionally (mask, skeleton) for visualisation)."""
    cfg = cfg or MeasureConfig()
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    gray = gray.astype(np.float32)
    h, w = gray.shape

    # 1. flatten illumination: cracks are darker than their local surroundings
    background = cv2.GaussianBlur(gray, (0, 0), cfg.flatten_sigma)
    flat = gray - background

    # 2. ridge response for dark lines
    ridge = sato(flat, sigmas=cfg.sato_sigmas, black_ridges=True)

    # 3. hysteresis threshold with an absolute + noise-adaptive floor
    floor = max(cfg.ridge_floor, cfg.ridge_median_mult * float(np.median(ridge)))
    if ridge.max() <= floor:
        metrics = _empty_metrics()
        if return_masks:
            z = np.zeros((h, w), dtype=bool)
            return metrics, z, z
        return metrics
    try:
        otsu = threshold_otsu(ridge)
    except ValueError:
        otsu = floor
    high = max(otsu, floor)
    low = max(cfg.low_ratio * high, 0.5 * floor)
    mask = apply_hysteresis_threshold(ridge, low, high)

    # 4. clean up
    mask = _remove_small(mask, cfg.min_area_px)
    skel = skeletonize(mask)

    labels, n = ndi.label(skel, structure=np.ones((3, 3)))
    if n > 0:
        sizes = ndi.sum(skel, labels, index=np.arange(1, n + 1))
        keep_ids = np.where(sizes >= cfg.min_skeleton_px)[0] + 1
        skel = np.isin(labels, keep_ids)
        n_components = int(len(keep_ids))
    else:
        n_components = 0

    if skel.sum() == 0:
        metrics = _empty_metrics()
        if return_masks:
            z = np.zeros((h, w), dtype=bool)
            return metrics, z, z
        return metrics

    # 5. width refinement. The ridge mask is wider than the crack (it inherits the filter
    #    scale), so re-measure width from the intensity profile: around each centreline pixel,
    #    take pixels darker than `width_depth_frac` x the local crack depth (a FWHM-style width).
    smooth = cv2.GaussianBlur(flat, (0, 0), 0.7)
    darkness = -smooth
    local_depth = ndi.maximum_filter(darkness, size=3)            # tolerate 1px centreline offset
    dist_to_skel, (iy, ix) = ndi.distance_transform_edt(~skel, return_indices=True)
    depth_map = local_depth[iy, ix]                               # nearest-centreline crack depth
    mask = (dist_to_skel <= cfg.width_search_radius) & (depth_map > 0) \
        & (darkness > cfg.width_depth_frac * depth_map)
    mask |= skel

    dist = ndi.distance_transform_edt(mask)
    widths = np.clip(2.0 * dist[skel] - 1.0, 1.0, None)
    metrics = {
        "length_px": float(skel.sum()),
        "mean_width_px": float(widths.mean()),
        "p90_width_px": float(np.percentile(widths, 90)),
        "max_width_px": float(widths.max()),
        "area_px": float(mask.sum()),
        "coverage": float(mask.sum()) / float(h * w),
        "n_components": n_components,
    }
    if return_masks:
        return metrics, mask, skel
    return metrics


def measure_crack_file(path: str, cfg: MeasureConfig = None):
    """Convenience wrapper for multiprocessing: returns metrics dict (all zeros if unreadable)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        m = _empty_metrics()
        m["read_error"] = 1
        return m
    m = measure_crack(img, cfg)
    m["read_error"] = 0
    return m


def overlay(img_bgr: np.ndarray, mask: np.ndarray, skel: np.ndarray) -> np.ndarray:
    """Crack mask in translucent yellow, skeleton in red, for human spot-checks."""
    out = img_bgr.copy()
    tint = out.copy()
    tint[mask] = (0, 255, 255)
    out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
    out[skel] = (0, 0, 255)
    return out
