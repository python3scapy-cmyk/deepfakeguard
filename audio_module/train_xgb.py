import argparse
import json
import os
import sys
import numpy as np
import xgboost as xgb
import kagglehub
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

try:
    from .feature_extractor import AudioFeatureExtractor
except ImportError:  # allow `python audio_module/train_xgb.py` direct run
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from audio_module.feature_extractor import AudioFeatureExtractor

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_audio.json")


def load_protocol_labels(la_root):
    """Parse ASVspoof2019 CM protocol files -> {file_id: 0 bonafide / 1 spoof}.
    Line format: SPEAKER_ID FILE_ID - SYSTEM_ID KEY  (KEY = bonafide|spoof)"""
    labels = {}
    proto_dir = os.path.join(la_root, "ASVspoof2019_LA_cm_protocols")
    for name in os.listdir(proto_dir):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(proto_dir, name)) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[-1].lower()
                if key in ("bonafide", "spoof"):
                    labels[parts[1]] = 1 if key == "spoof" else 0
    return labels


def collect_split(la_root, split_dir, labels, extractor, max_per_class=None):
    """Extract features for every labeled flac in one split directory."""
    cache = os.path.join(os.path.dirname(MODEL_PATH),
                         f"feats_{split_dir}_{max_per_class or 'all'}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        print(f"[INFO] {split_dir}: loaded {len(d['y'])} cached feature vectors")
        return list(d["X"]), list(d["y"])
    flac_dir = os.path.join(la_root, split_dir, "flac")
    X, y = [], []
    counts = {0: 0, 1: 0}
    files = sorted(os.listdir(flac_dir)) if os.path.isdir(flac_dir) else []
    print(f"[INFO] {split_dir}: {len(files)} files on disk")
    for i, fname in enumerate(files):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in (".flac", ".wav"):
            continue
        label = labels.get(stem)
        if label is None:
            continue  # not in protocol files -> no ground truth -> skip
        if max_per_class and counts[label] >= max_per_class:
            continue
        feats = extractor.extract_features(os.path.join(flac_dir, fname))
        if feats is not None:
            X.append(feats)
            y.append(label)
            counts[label] += 1
        if (i + 1) % 1000 == 0:
            print(f"[INFO]   ...{i + 1}/{len(files)} processed "
                  f"(bonafide={counts[0]} spoof={counts[1]})")
    print(f"[INFO] {split_dir}: kept bonafide={counts[0]} spoof={counts[1]}")
    np.savez(cache, X=np.array(X), y=np.array(y))
    return X, y


def start_training(max_per_class=None):
    extractor = AudioFeatureExtractor()
    print("[INFO] Checking audio dataset via kagglehub...")
    download_path = kagglehub.dataset_download("radhaparikh/asvspoof2019-subset")
    print(f"[SUCCESS] Dataset located at: {download_path}")

    # Find the LA root (contains ASVspoof2019_LA_cm_protocols)
    la_root = None
    for root, dirs, files in os.walk(download_path):
        if "ASVspoof2019_LA_cm_protocols" in dirs:
            la_root = root
            break
    if la_root is None:
        sys.exit("[FATAL] Could not find ASVspoof2019_LA_cm_protocols in the "
                 "downloaded dataset -- layout unexpected.")

    labels = load_protocol_labels(la_root)
    n_bona = sum(1 for v in labels.values() if v == 0)
    print(f"[INFO] Protocol labels loaded: {len(labels)} file IDs "
          f"(bonafide={n_bona}, spoof={len(labels) - n_bona})")

    # Official splits: train on LA_train, evaluate on LA_dev
    X_train, y_train = collect_split(la_root, "ASVspoof2019_LA_train",
                                     labels, extractor, max_per_class)
    X_test, y_test = collect_split(la_root, "ASVspoof2019_LA_dev",
                                   labels, extractor, max_per_class)

    for name, yy in (("train", y_train), ("dev", y_test)):
        if len(yy) == 0 or len(np.unique(yy)) < 2:
            sys.exit(f"[FATAL] {name} split empty or single-class "
                     f"(n={len(yy)}, classes={np.unique(yy).tolist()}). "
                     "Refusing to train on synthetic data.")

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Class weighting: spoof outnumbers bonafide in ASVspoof
    spw = float((y_train == 0).sum()) / max(1, (y_train == 1).sum())
    model = xgb.XGBClassifier(eval_metric="logloss", scale_pos_weight=spw)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    print(f"[SUCCESS] XGBoost trained on official LA_train split.")
    print(f"[METRICS] LA_dev accuracy: {acc:.3f} ({len(y_test)} samples)")
    print(f"[METRICS] Spoof recall (spoofs caught):    {tp / (tp + fn):.3f}")
    print(f"[METRICS] Bonafide recall (real accepted): {tn / (tn + fp):.3f}")

    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thr = roc_curve(y_test, proba)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    eer, eer_thr = (fpr[i] + fnr[i]) / 2, float(thr[i])
    print(f"[METRICS] EER: {eer:.3f} at threshold {eer_thr:.3f}")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        pred = (proba >= t).astype(int)
        sr = (pred[y_test == 1] == 1).mean()
        br = (pred[y_test == 0] == 0).mean()
        print(f"[SWEEP] thr={t:.1f}  spoof_recall={sr:.3f}  bonafide_recall={br:.3f}")

    model.save_model(MODEL_PATH)
    # Save calibration sidecar: the detector remaps raw probabilities so the
    # EER threshold lands at 0.5, giving the fusion a balanced signal.
    calib_path = MODEL_PATH.replace(".json", "_calibration.json")
    with open(calib_path, "w") as f:
        json.dump({"eer": round(float(eer), 4),
                   "eer_threshold": round(eer_thr, 4)}, f)
    print(f"[SUCCESS] Model saved to {MODEL_PATH} -- AASISTDetector will "
          "auto-load it on next run.")
    print(f"[SUCCESS] Calibration saved to {calib_path} "
          f"(EER threshold {eer_thr:.3f} -> 0.5)")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="cap samples per class per split (quick test runs)")
    args = ap.parse_args()
    start_training(max_per_class=args.max_per_class)
