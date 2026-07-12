#!/usr/bin/env python3
"""
P0-4: visual deepfake detector evaluation.

Measures how well the detector separates REAL photos from AI/deepfake
photos, and finds the best decision threshold.

Setup - two folders of test images:
    eval_data/
      real/   <- your own webcam selfies, teammates' photos (10-20 images)
      fake/   <- AI faces: thispersondoesnotexist.com saves, midjourney/
                 stable-diffusion portraits, faceswap stills (10-20 images)

Run:
    python detector_eval.py --real eval_data/real --fake eval_data/fake

Output: per-file probabilities, accuracy at 0.5, and a threshold sweep so
you can see where the real separation point is for THIS model on YOUR data.
"""
import argparse
import os
import sys

import cv2
import numpy as np

from vision.deepfake_detector import DeepfakeDetector

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def score_folder(det, folder, label):
    rows = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() not in IMG_EXT:
            continue
        frame = cv2.imread(os.path.join(folder, name))
        if frame is None:
            print(f"  [skip] {name}: could not decode")
            continue
        r = det.analyze_single_frame(frame)
        if not r["face_found"]:
            print(f"  [skip] {name}: no face found")
            continue
        p = r["deepfake_probability"]
        rows.append((name, p))
        print(f"  {label:<4} {name:<40} fake_prob={p:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="folder of real photos")
    ap.add_argument("--fake", required=True, help="folder of AI/deepfake photos")
    args = ap.parse_args()

    det = DeepfakeDetector()
    if det.backend is None:
        print("\n[FATAL] Real classifier not loaded (mock mode) - eval is "
              "meaningless. Fix model download first.")
        sys.exit(1)
    print(f"\nBackend: {det.backend}\n")

    print("REAL folder:")
    real_rows = score_folder(det, args.real, "REAL")
    print("\nFAKE folder:")
    fake_rows = score_folder(det, args.fake, "FAKE")

    if not real_rows or not fake_rows:
        print("\n[FATAL] Need at least one scored image in each folder.")
        sys.exit(1)

    real_p = np.array([p for _, p in real_rows])
    fake_p = np.array([p for _, p in fake_rows])

    print("\n" + "=" * 62)
    print(f"  REAL: n={len(real_p)}  mean={real_p.mean():.3f}  "
          f"min={real_p.min():.3f}  max={real_p.max():.3f}")
    print(f"  FAKE: n={len(fake_p)}  mean={fake_p.mean():.3f}  "
          f"min={fake_p.min():.3f}  max={fake_p.max():.3f}")

    print("\n  Threshold sweep (verdict FAKE when prob > t):")
    print(f"  {'t':>5} | {'real acc':>8} | {'fake acc':>8} | {'balanced':>8}")
    best_t, best_bal = 0.5, 0.0
    for t in np.arange(0.20, 0.85, 0.05):
        ra = float((real_p <= t).mean())   # real correctly kept REAL
        fa = float((fake_p > t).mean())    # fake correctly flagged FAKE
        bal = (ra + fa) / 2
        marker = ""
        if bal > best_bal:
            best_bal, best_t, marker = bal, t, "  <- best"
        print(f"  {t:>5.2f} | {ra:>8.2%} | {fa:>8.2%} | {bal:>8.2%}{marker}")

    print("\n  Suggested threshold: {:.2f} (balanced accuracy {:.1%})"
          .format(best_t, best_bal))
    print("  At the default 0.50: real acc {:.1%}, fake acc {:.1%}"
          .format(float((real_p <= 0.5).mean()), float((fake_p > 0.5).mean())))
    print("=" * 62)
    print("\nIf separation is poor (means closer than ~0.25), tell Claude -")
    print("the next step is swapping the HF checkpoint, not tuning thresholds.")


if __name__ == "__main__":
    main()
