"""
Week 3 audio anti-spoofing: AASIST + XGBoost ensemble.

If librosa is installed, feature extraction runs for real; the
AASIST/XGBoost *scoring* itself is mocked (heuristic on the audio's
own statistics) unless real model checkpoints are wired in later --
this keeps the payload contract and RTF/latency instrumentation
identical to what the real models would eventually produce.
"""
import time
from collections import deque

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


class AASISTDetector:
    def __init__(self, model_path=None, device="cpu"):
        self.device = device
        self.sample_rate = 16000
        self.chunk_duration = 1.0
        self.window = deque(maxlen=10)

    def _mock_aasist_score(self, audio_chunk):
        energy = np.std(audio_chunk)
        score = 0.3 + 0.4 * (1.0 - min(1.0, energy / 0.1))
        score += np.random.normal(0, 0.05)
        return float(np.clip(score, 0.0, 1.0))

    def _mock_xgboost_score(self, audio_chunk):
        if _HAS_LIBROSA:
            try:
                mfcc = librosa.feature.mfcc(y=audio_chunk, sr=self.sample_rate, n_mfcc=13)
                mfcc_std = float(np.std(mfcc, axis=1).mean())
                score = 0.4 + 0.3 * (1.0 - min(1.0, mfcc_std / 10.0))
            except Exception:
                score = 0.4
        else:
            score = 0.4
        score += np.random.normal(0, 0.05)
        return float(np.clip(score, 0.0, 1.0))

    def ensemble_score(self, audio_chunk):
        start = time.time()
        expected_len = int(self.sample_rate * self.chunk_duration)
        if len(audio_chunk) < expected_len:
            audio_chunk = np.pad(audio_chunk, (0, expected_len - len(audio_chunk)))
        else:
            audio_chunk = audio_chunk[:expected_len]

        aasist = self._mock_aasist_score(audio_chunk)
        xgb = self._mock_xgboost_score(audio_chunk)
        final = 0.70 * aasist + 0.30 * xgb

        self.window.append(final)
        smoothed = float(np.mean(self.window)) if self.window else final

        latency = time.time() - start
        rtf = latency / self.chunk_duration

        return {
            "spoof_probability": smoothed,
            "model": "aasist_xgboost_ensemble",
            "aasist_raw": aasist,
            "xgboost_raw": xgb,
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
