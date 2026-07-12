"""
Lip-sync verification via a correlation proxy: mouth-openness (from
video) vs audio RMS energy, correlated over a rolling window. This
replaces a full SyncNet model with something lightweight enough to
run in real time; swap in a real SyncNet later without changing the
payload shape.
"""
import time
from collections import deque
from typing import Any, Dict, Optional

import cv2
import numpy as np


class LipSyncDetector:
    def __init__(self, window_ms: int = 200):
        self.window_ms = window_ms
        self.mouth_openness_buffer = deque(maxlen=50)
        self.audio_rms_buffer = deque(maxlen=50)
        self.sync_score = 0.5
        self.threshold = 0.3
        self.last_update = time.time()

    def update_mouth_openness(self, mouth_crop: Optional[np.ndarray]) -> None:
        if mouth_crop is None or mouth_crop.size == 0:
            self.mouth_openness_buffer.append(0.0)
            return
        try:
            gray = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY) if len(mouth_crop.shape) == 3 else mouth_crop
            _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            openness = np.sum(thresh > 0) / thresh.size
            self.mouth_openness_buffer.append(float(openness))
        except Exception:
            self.mouth_openness_buffer.append(0.0)

    def update_audio(self, audio_chunk: np.ndarray) -> None:
        if audio_chunk is None or len(audio_chunk) == 0:
            self.audio_rms_buffer.append(0.0)
            return
        rms = float(np.sqrt(np.mean(np.square(audio_chunk))))
        self.audio_rms_buffer.append(rms)

    def compute_sync_score(self) -> Dict[str, Any]:
        if len(self.mouth_openness_buffer) < 5 or len(self.audio_rms_buffer) < 5:
            return {
                "lip_sync_score": 0.5,
                "in_sync": False,
                "method": "correlation_proxy",
                "window_ms": self.window_ms,
                "latency_ms": 0.0,
                "confidence": "insufficient_data",
            }

        # Always use the SAME number of samples from both buffers, taken
        # from the end of each -- this is what earlier versions got wrong
        # (mismatched lengths -> np.corrcoef ValueError).
        use_len = max(5, min(len(self.mouth_openness_buffer), len(self.audio_rms_buffer), 30))
        mouth = np.array(list(self.mouth_openness_buffer)[-use_len:], dtype=np.float64)
        audio = np.array(list(self.audio_rms_buffer)[-use_len:], dtype=np.float64)

        mouth_std = np.std(mouth)
        audio_std = np.std(audio)
        if mouth_std < 1e-10 or audio_std < 1e-10:
            correlation = 0.0
        else:
            mouth_norm = (mouth - np.mean(mouth)) / mouth_std
            audio_norm = (audio - np.mean(audio)) / audio_std
            correlation = float(np.mean(mouth_norm * audio_norm))  # manual Pearson r
            if np.isnan(correlation) or np.isinf(correlation):
                correlation = 0.0

        self.sync_score = float(max(0.0, min(1.0, (correlation + 1.0) / 2.0)))
        confidence = "high" if use_len >= 20 else "medium" if use_len >= 10 else "low"

        return {
            "lip_sync_score": round(self.sync_score, 3),
            "in_sync": bool(self.sync_score > self.threshold),
            "method": "correlation_proxy",
            "window_ms": self.window_ms,
            "latency_ms": round((time.time() - self.last_update) * 1000, 1),
            "confidence": confidence,
            "raw_correlation": round(correlation, 3),
            "samples_used": use_len,
        }

    def reset(self):
        self.mouth_openness_buffer.clear()
        self.audio_rms_buffer.clear()
        self.sync_score = 0.5
