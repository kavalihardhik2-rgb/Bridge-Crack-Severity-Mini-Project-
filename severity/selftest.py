"""
Sanity test for measure_cracks.py using synthetic images (no dataset required).

Checks:
  * a blank textured image yields (almost) no crack
  * wider synthetic cracks give larger measured width
  * longer synthetic cracks give larger measured length
Run:  python severity/selftest.py
"""
import cv2
import numpy as np

try:
    from severity.measure_cracks import measure_crack
except ImportError:  # running as a plain script from inside severity/
    from measure_cracks import measure_crack


def synth(thickness=0, length_frac=0.0, seed=0, size=256):
    rng = np.random.RandomState(seed)
    base = 150 + rng.randn(size, size).astype(np.float32) * 6      # concrete-like noise
    base = cv2.GaussianBlur(base, (0, 0), 1.0)
    img = np.clip(base, 0, 255).astype(np.uint8)
    if thickness > 0:
        n_pts = 12
        x_end = int(20 + length_frac * (size - 40))
        xs = np.linspace(20, x_end, n_pts)
        ys = size // 2 + np.cumsum(rng.randn(n_pts) * 6)
        pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, 40, thickness=thickness, lineType=cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main():
    blank = measure_crack(synth(0))
    print("blank      :", {k: round(v, 2) for k, v in blank.items()})

    widths = []
    for t in (1, 3, 6):
        m = measure_crack(synth(t, 1.0, seed=1))
        widths.append(m["mean_width_px"])
        print(f"thickness {t}: length={m['length_px']:.0f}  mean_w={m['mean_width_px']:.2f}  "
              f"p90_w={m['p90_width_px']:.2f}  coverage={m['coverage']:.4f}")

    lengths = []
    for lf in (0.3, 0.6, 1.0):
        m = measure_crack(synth(3, lf, seed=2))
        lengths.append(m["length_px"])
        print(f"length_frac {lf}: length={m['length_px']:.0f}")

    assert blank["length_px"] < 30, "blank texture should produce (almost) no crack"
    assert widths[0] < widths[1] < widths[2], f"width should increase with thickness: {widths}"
    assert lengths[0] < lengths[1] < lengths[2], f"length should increase with extent: {lengths}"
    print("\nOK: width and length measurements respond monotonically on synthetic cracks.")


if __name__ == "__main__":
    main()
