"""
Week 3 audio anti-spoofing -- REAL scoring path.

Backend priority:
  1. "aasist_real"    -- AASIST checkpoint (audio_module/models/AASIST.pth)
  2. "xgboost_real"   -- XGBoost/MFCC model (audio_module/models/xgb_audio.json)
  3. "mock"           -- heuristic, LOUDLY flagged. Never silent.
"""
import os
import sys
import time
from collections import deque

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

AASIST_SAMPLES = 64600
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
AASIST_REPO = os.path.join(_ROOT, "third_party", "aasist")
AASIST_CKPT = os.path.join(_HERE, "models", "AASIST.pth")
XGB_MODEL = os.path.join(_HERE, "models", "xgb_audio.json")


class AASISTDetector:
    def __init__(self, model_path=None, device="cpu", allow_mock=True):
        self.device = device
        self.sample_rate = 16000
        self.chunk_duration = 4.0
        self.window = deque(maxlen=10)

        self.backend = None
        self.model = None
        self.xgb = None
        self.extractor = None

        if self._load_aasist(model_path or AASIST_CKPT):
            self.backend = "aasist_real"
        elif self._load_xgboost():
            self.backend = "xgboost_real"
        elif allow_mock:
            self.backend = "mock"
            print("\n" + "*" * 64)
            print("*** RUNNING WITH MOCK AUDIO DETECTOR -- NOT DEMO-READY ***")
            print("*" * 64 + "\n")
        else:
            raise RuntimeError("No real audio model available and allow_mock=False.")

    def _load_aasist(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            return False
        if not os.path.isdir(AASIST_REPO):
            print(f"[WARN] {AASIST_REPO} missing")
            return False
        try:
            import json
            import torch
            if AASIST_REPO not in sys.path:
                sys.path.insert(0, AASIST_REPO)
            from models.AASIST import Model

            conf = json.load(open(os.path.join(AASIST_REPO, "config", "AASIST.conf")))
            self.model = Model(conf["model_config"])
            state = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device).eval()
            self._torch = torch
            n = sum(p.numel() for p in self.model.parameters())
            print(f"[INFO] Real AASIST loaded ({n:,} params)")
            return True
        except Exception as e:
            print(f"[WARNING] AASIST load failed ({e}) -> trying XGBoost")
            self.model = None
            return False

    def _load_xgboost(self):
        if not os.path.exists(XGB_MODEL):
            return False
        try:
            import xgboost as xgb
            from .feature_extractor import AudioFeatureExtractor
            self.xgb = xgb.XGBClassifier()
            self.xgb.load_model(XGB_MODEL)
            self.extractor = AudioFeatureExtractor(sample_rate=self.sample_rate)
            print(f"[INFO] Real XGBoost audio model loaded")
            return True
        except Exception as e:
            print(f"[WARNING] XGBoost load failed ({e}) -> mock")
            self.xgb = None
            return False

    def _score_aasist(self, audio_chunk):
        torch = self._torch
        x = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            logits = out[1] if isinstance(out, (tuple, list)) else out
            probs = torch.softmax(logits, dim=1)
            bonafide = probs[0, 1].item()
        return float(np.clip(1.0 - bonafide, 0.0, 1.0))

    def _score_xgboost(self, audio_chunk):
        feats = self.extractor.extract_features_from_array(audio_chunk, self.sample_rate)
        if feats is None:
            return 0.5
        prob = self.xgb.predict_proba(feats.reshape(1, -1))[0, 1]
        return float(np.clip(prob, 0.0, 1.0))

    def _score_mock(self, audio_chunk):
        energy = np.std(audio_chunk)
        score = 0.3 + 0.4 * (1.0 - min(1.0, energy / 0.1))
        score += np.random.normal(0, 0.05)
        return float(np.clip(score, 0.0, 1.0))

    def ensemble_score(self, audio_chunk):
        start = time.time()
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

        need = AASIST_SAMPLES if self.backend == "aasist_real" else int(
            self.sample_rate * self.chunk_duration)
        if len(audio_chunk) < need:
            audio_chunk = np.pad(audio_chunk, (0, need - len(audio_chunk)))
        else:
            audio_chunk = audio_chunk[-need:]

        if self.backend == "aasist_real":
            spoof = self._score_aasist(audio_chunk)
        elif self.backend == "xgboost_real":
            spoof = self._score_xgboost(audio_chunk)
        else:
            spoof = self._score_mock(audio_chunk)

        self.window.append(spoof)
        smoothed = float(np.mean(self.window))

        latency = time.time() - start
        duration = len(audio_chunk) / self.sample_rate
        rtf = latency / duration if duration > 0 else 0.0

        return {
            "spoof_probability": smoothed,
            "spoof_probability_raw": spoof,
            "model": self.backend,
            "model_backend": self.backend,
            "rtf": round(rtf, 3),
            "latency_ms": round(latency * 1000, 1),
            "rms_level": float(np.sqrt(np.mean(audio_chunk ** 2))),
            "rms_level": float(np.sqrt(np.mean(audio_chunk ** 2))),
            "rms_level": float(np.sqrt(np.mean(audio_chunk ** 2))),
            "audio_quality": "clean" if np.std(audio_chunk) > 0.01 else "silent_or_noisy",
            "timestamp": time.time(),
        }

    def get_audio_spoof_score(self):
        if not self.window:
            return 50
        return int(np.mean(self.window) * 100)

    def reset(self):
        self.window.clear()
