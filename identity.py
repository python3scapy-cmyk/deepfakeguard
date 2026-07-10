"""
Importable ArcFace 1:1 identity matching, extracted from verify.py's core
logic so main.py can call it directly instead of running verify.py as a
separate process with its own competing cv2.VideoCapture(0).

verify.py and enroll.py are left in place as standalone CLI tools (enrollment
is still a deliberate, explicit step a user runs once via `python -m
security.enroll`), but the actual per-frame scoring function now lives here
so it can be shared.
"""
import json
import os
import time

import numpy as np

DEFAULT_ENROLLED_PATH = os.path.join(os.path.dirname(__file__), "enrolled_face.json")


def _has_faces(faces):
    if faces is None:
        return False
    if isinstance(faces, np.ndarray):
        return faces.size > 0
    return len(faces) > 0


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class IdentityMatcher:
    """Loads an enrolled face embedding once and scores live frames against
    it using InsightFace/ArcFace. Falls back gracefully (rather than
    crashing main.py) if InsightFace can't be loaded or nothing has been
    enrolled yet -- both are treated as "identity not verified", not a
    silent 85/100 pass."""

    # If nothing is enrolled, identity can't be confirmed -- score low
    # rather than defaulting to a value that would read as "trusted".
    NOT_ENROLLED_SCORE = 20

    def __init__(self, enrolled_path=DEFAULT_ENROLLED_PATH, threshold=0.6, det_size=(320, 320)):
        self.threshold = threshold
        self.enrolled_embedding = None
        self.enrolled = False
        self.app = None
        self._last_score = self.NOT_ENROLLED_SCORE

        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=det_size)
        except Exception as e:
            print(f"[WARNING] Could not load InsightFace ({e}) -- identity "
                  f"matching disabled, scoring will report 'not enrolled'.")
            self.app = None

        if os.path.exists(enrolled_path):
            try:
                with open(enrolled_path, "r") as f:
                    data = json.load(f)
                self.enrolled_embedding = np.array(data["embedding"])
                self.enrolled = True
            except Exception as e:
                print(f"[WARNING] Could not load enrolled face from {enrolled_path}: {e}")
        else:
            print(f"[WARNING] No enrolled face found at {enrolled_path}. "
                  f"Run `python -m security.enroll` first, or identity_score "
                  f"will stay at {self.NOT_ENROLLED_SCORE}/100.")

    def last_score(self):
        """0-100 identity score for frames where we don't re-run inference
        (main.py only calls compute_identity_score every ~15th frame)."""
        return self._last_score

    def compute_identity_score(self, frame, faces=None):
        """Runs ArcFace on `frame` and compares to the enrolled embedding.

        Returns a dict shaped like the identity module's documented
        contract (identity_score, similarity_score, confidence, etc.) so it
        can be dropped straight into main.py's fusion payload.
        """
        start = time.time()

        if self.app is None or not self.enrolled:
            result = {
                "identity_score": self.NOT_ENROLLED_SCORE,
                "similarity_score": 0.0,
                "threshold_used": self.threshold,
                "face_detected": _has_faces(faces),
                "multiple_faces_detected": _has_faces(faces) and len(faces) > 1,
                "confidence": "none",
                "latency_ms": round((time.time() - start) * 1000, 1),
                "verdict": "FAKE",
                "reason": "not_enrolled" if self.app is not None else "insightface_unavailable",
            }
            self._last_score = result["identity_score"]
            return result

        detected = self.app.get(frame)

        if len(detected) == 0:
            result = {
                "identity_score": 0,
                "similarity_score": 0.0,
                "threshold_used": self.threshold,
                "face_detected": False,
                "multiple_faces_detected": False,
                "confidence": "none",
                "latency_ms": round((time.time() - start) * 1000, 1),
                "verdict": "FAKE",
            }
        elif len(detected) > 1:
            result = {
                "identity_score": 0,
                "similarity_score": 0.0,
                "threshold_used": self.threshold,
                "face_detected": True,
                "multiple_faces_detected": True,
                "confidence": "none",
                "latency_ms": round((time.time() - start) * 1000, 1),
                "verdict": "FAKE",
            }
        else:
            live_embedding = detected[0].normed_embedding
            score = round(cosine_similarity(live_embedding, self.enrolled_embedding), 3)
            if score >= self.threshold:
                confidence = "high"
            elif score >= 0.50:
                confidence = "low"
            else:
                confidence = "none"
            result = {
                "identity_score": round(max(0.0, min(1.0, score)) * 100),
                "similarity_score": score,
                "threshold_used": self.threshold,
                "face_detected": True,
                "multiple_faces_detected": False,
                "confidence": confidence,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "verdict": "REAL" if score >= self.threshold else "FAKE",
            }

        self._last_score = result["identity_score"]
        return result
