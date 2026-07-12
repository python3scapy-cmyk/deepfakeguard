"""
Week 3->4 audio anti-spoofing.

Runtime scoring is now a REAL XGBoost classifier trained on an ASVspoof
2019 subset (see train_xgb.py, which saves xgb_audio.json next to this
file). The detector auto-loads that checkpoint on startup and runs true
predict_proba inference on 26-dim MFCC+delta features -- the exact same
feature pipeline used at training time (AudioFeatureExtractor).

If the checkpoint or librosa is missing, the detector DOES NOT silently
fabricate scores: it prints a loud warning, and every payload it emits is
labeled model="mock_heuristic" so the dashboard/eval can never mistake
heuristic output for real inference.

Class name kept as AASISTDetector so main.py's import contract is
untouched; the payload's "model" field states what actually ran.
"""
import json
import os
import time
from collections import deque

import numpy as np

try:
    import librosa  # noqa: F401  (needed by AudioFeatureExtractor)
    from .feature_extractor import AudioFeatureExtractor
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_audio.json")


class AASISTDetector:
    def __init__(self, model_path=None, device="cpu"):
        self.device = device
        self.sample_rate = 16000
        self.chunk_duration = 1.0
        self.window = deque(maxlen=10)

        self.model = None
        self.model_name = "mock_heuristic"
        self.extractor = AudioFeatureExtractor(
            sample_rate=self.sample_rate) if _HAS_LIBROSA else None

        # Calibration: raw XGBoost probabilities are spoof-skewed (EER sits
        # near 0.9, not 0.5). We piecewise-linearly remap so the EER threshold
        # maps to 0.5, giving the fusion a balanced, interpretable signal.
        self.calib_threshold = 0.5  # identity remap unless sidecar found

        path = model_path or _DEFAULT_MODEL_PATH
        if not _HAS_LIBROSA:
            print("[WARNING] librosa not installed -> audio spoof scoring "
                  "falls back to mock heuristic (clearly labeled in payload).")
        elif os.path.exists(path):
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier()
                self.model.load_model(path)
                self.model_name = "xgboost_asvspoof2019"
                print(f"[INFO] Real XGBoost audio spoof model loaded from {path}")
                calib_path = path.replace(".json", "_calibration.json")
                if os.path.exists(calib_path):
                    with open(calib_path) as cf:
                        self.calib_threshold = float(
                            json.load(cf).get("eer_threshold", 0.5))
                    print(f"[INFO] Calibration loaded: EER threshold "
                          f"{self.calib_threshold:.3f} -> 0.5")
                else:
                    print("[WARNING] No calibration sidecar "
                          f"({calib_path}) -- raw probabilities used. "
                          "Re-run train_xgb to generate it.")
            except Exception as e:
                print(f"[WARNING] Could not load XGBoost checkpoint ({e}) -> "
                      "mock heuristic fallback (clearly labeled).")
        else:
            print(f"[WARNING] No trained model at {path}. Run "
                  "`python -m audio_module.train_xgb` once to create it. "
                  "Falling back to mock heuristic (clearly labeled).")

    # ---------------- scoring backends ----------------
    def _calibrate(self, p):
        """Piecewise-linear remap so calib_threshold -> 0.5, endpoints fixed."""
        t = self.calib_threshold
        if t <= 0.0 or t >= 1.0 or t == 0.5:
            return p
        if p <= t:
            return 0.5 * p / t
        return 0.5 + 0.5 * (p - t) / (1.0 - t)

    def _real_xgb_score(self, audio_chunk):
        """Calibrated P(spoof) from the trained classifier on the live chunk."""
        feats = self.extractor.extract_features_from_array(audio_chunk)
        proba = self.model.predict_proba(feats.reshape(1, -1))[0]
        return self._calibrate(float(proba[1]))  # class 1 == spoof

    def _mock_heuristic_score(self, audio_chunk):
        """Last-resort heuristic. Only used when no real model is available,
        and always labeled model='mock_heuristic' in the payload."""
        energy = np.std(audio_chunk)
        score = 0.35 + 0.3 * (1.0 - min(1.0, energy / 0.1))
        return float(np.clip(score, 0.0, 1.0))

    # ---------------- public API (contract unchanged) ----------------
    def ensemble_score(self, audio_chunk):
        start = time.time()
        expected_len = int(self.sample_rate * self.chunk_duration)
        if len(audio_chunk) < expected_len:
            audio_chunk = np.pad(audio_chunk, (0, expected_len - len(audio_chunk)))
        else:
            audio_chunk = audio_chunk[:expected_len]
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

        if self.model is not None:
            raw = self._real_xgb_score(audio_chunk)
        else:
            raw = self._mock_heuristic_score(audio_chunk)

        self.window.append(raw)
        smoothed = float(np.mean(self.window)) if self.window else raw

        latency = time.time() - start
        rtf = latency / self.chunk_duration

        return {
            "spoof_probability": smoothed,
            "model": self.model_name,
            "xgboost_raw": raw,
            "rtf": round(rtf, 3),
            "latency_ms": round(latency * 1000, 1),
            "audio_quality": "clean" if np.std(audio_chunk) > 0.01 else "noisy",
            "timestamp": time.time(),
        }

    def get_audio_spoof_score(self):
        """0-100 scale for the fusion engine (higher = more likely spoofed)."""
        if not self.window:
            return 50
        return int(np.mean(self.window) * 100)

    def reset(self):
        self.window.clear()
