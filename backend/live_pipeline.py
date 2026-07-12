"""
Live browser-camera pipeline.

main.py opens an OS camera directly (cv2.VideoCapture) and runs every
frame through vision.face_detector, vision.liveness_engine,
vision.deepfake_detector and security.identity locally, then POSTs the
fused result to this backend's /fusion endpoint. A browser page can't
hand its getUserMedia frames to a local Python process the same way --
there's no direct pipe between JS and this server's OpenCV code. So
instead, participant.html grabs periodic JPEG snapshots from its own
camera preview and POSTs them to /live/frame (see backend/app.py); this
module runs those snapshots through the IDENTICAL detector stack
main.py uses, and produces the same fusion JSON shape main.py's
generate_json() does, so it can be ingested by the exact same
_ingest_fusion_payload() code path as POST /fusion.

Scope (v1): visual signals only -- identity match, liveness challenge,
visual deepfake probability, and injection/virtual-camera risk. Audio
(audio_spoof_probability, lip_sync_score) is intentionally left out:
main.py's audio path reads from a local microphone via `sounddevice`
in a fixed 16kHz float32 format, which isn't something a browser hands
over the same way a JPEG frame does. A follow-up /live/audio endpoint
could add it later without changing anything here.

Like the rest of this backend (one global latest_scores / session_log
/ EMA state), this assumes ONE active demo session at a time -- a
single shared LiveVisionSession instance, not one per session_id.
/session-reset clears it exactly like it clears everything else.
"""
import base64
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np

from vision.face_detector import FaceLandmarkDetector
from vision.liveness_engine import LivenessChallengeEngine
from vision.deepfake_detector import DeepfakeDetector
from security.identity import IdentityMatcher
from security.device_check import is_virtual_camera
from security.timing_check import normalize_anomaly_score


def decode_frame(image_data_url):
    """Decodes a data-URL (or bare base64) JPEG/PNG string, as produced by
    canvas.toDataURL() in the browser, into a BGR frame. Returns None on
    any decode failure rather than raising, so a single malformed upload
    can't take down the live pipeline."""
    if not image_data_url or not isinstance(image_data_url, str):
        return None
    b64 = image_data_url.split(",", 1)[1] if "," in image_data_url else image_data_url
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class LiveVisionSession:
    """Mirrors main.py's DeepfakeGuardSystem per-frame vision logic
    (challenge issuing, spoof-flag smoothing, hard-deny rules), but driven
    by discrete browser-uploaded frames arriving one HTTP request at a
    time instead of a cv2.VideoCapture loop. All the underlying detector
    classes (FaceLandmarkDetector, LivenessChallengeEngine,
    DeepfakeDetector, IdentityMatcher) are themselves already written as
    one-frame-at-a-time state machines, so no changes were needed there --
    only something to call them in sequence and remember state between
    HTTP requests, which is what this class does.
    """

    IDENTITY_EVERY_N_FRAMES = 5    # ArcFace is comparatively expensive
    CHALLENGE_COOLDOWN_FRAMES = 20  # frames of quiet before issuing a new challenge

    def __init__(self):
        self._build()

    def _build(self):
        self.face_detector = FaceLandmarkDetector()
        self.liveness_engine = LivenessChallengeEngine()
        self.deepfake_detector = DeepfakeDetector()
        self.identity_matcher = IdentityMatcher()

        self.frame_count = 0
        self.challenge_active = False
        self.challenge_cooldown = 0
        self.spoof_history = deque(maxlen=15)
        self.frame_arrival_sec = deque(maxlen=50)
        self.last_identity_result = None

    def reset(self):
        """Rebuilds all detector/state-machine instances from scratch. Fast
        enough to call on every dashboard "session reset" click -- the
        heavy models (identity's InsightFace, the deepfake classifier)
        reload from local/cached weights, not from a network round trip
        each time, and a reset is a rare explicit action, not a hot path."""
        self._build()

    def _frame_timing_anomaly(self, client_ts_ms):
        """Same signal security/timing_check.py computes (variance of
        inter-frame arrival time, normalized so a very regular/low-jitter
        stream -- characteristic of a replayed or virtual-camera feed --
        scores as more anomalous) but computed from browser-reported
        capture timestamps instead of opening a second local
        VideoCapture. The browser already holds the only handle on the
        real device, so re-opening it server-side isn't possible here --
        and isn't necessary, since inter-arrival timing is observable
        either way."""
        if client_ts_ms is None:
            return None
        self.frame_arrival_sec.append(float(client_ts_ms) / 1000.0)
        if len(self.frame_arrival_sec) < 10:
            return None
        deltas = np.diff(np.array(self.frame_arrival_sec))
        deltas = deltas[deltas > 0]
        if len(deltas) < 5:
            return None
        variance = float(np.var(deltas))
        return normalize_anomaly_score(variance)

    def process_frame(self, frame, session_id, client_ts_ms=None, device_name=None):
        """Runs one browser-uploaded frame through the full vision stack
        and returns a fusion payload shaped exactly like main.py's
        generate_json() output, ready for _ingest_fusion_payload()."""
        self.frame_count += 1

        faces, vision_payload = self.face_detector.process_frame(frame)
        _, liveness_payload = self.liveness_engine.process_frame(frame, faces)
        deepfake_result = self.deepfake_detector.score_frame(frame)

        identity_result = self.last_identity_result
        if self.frame_count % self.IDENTITY_EVERY_N_FRAMES == 0:
            identity_result = self.identity_matcher.compute_identity_score(frame, faces)
            self.last_identity_result = identity_result

        # ---- challenge issuing: same state machine as main.py's run() ----
        challenge = liveness_payload.get("challenge_result")
        if challenge:
            self.challenge_active = False
            self.challenge_cooldown = self.frame_count + self.CHALLENGE_COOLDOWN_FRAMES
            self.liveness_engine.reset()
        elif not self.challenge_active and self.frame_count > self.challenge_cooldown:
            if liveness_payload.get("face_detected") and liveness_payload.get("state") == "IDLE":
                self.liveness_engine.issue_challenge()
                self.challenge_active = True

        # ---- temporal smoothing of the 2D spoof flag (majority-vote over
        # recent frames, not a single noisy frame) ----
        spoof_analysis = vision_payload.get("spoof_analysis") or {}
        raw_spoof_flag = spoof_analysis.get("is_spoof", False) or liveness_payload.get("is_static", False)
        self.spoof_history.append(raw_spoof_flag)
        spoof_ratio = sum(self.spoof_history) / len(self.spoof_history)
        anti_spoof_2d_flag = spoof_ratio >= 0.6 and len(self.spoof_history) >= 5

        # ---- injection / virtual camera ----
        anomaly_score = self._frame_timing_anomaly(client_ts_ms)
        injection_risk_score = round((1 - anomaly_score) * 100) if anomaly_score is not None else 50
        virtual_camera_detected = False
        if device_name:
            virtual_camera_detected, _matched = is_virtual_camera(device_name)

        face_detected = liveness_payload.get("face_detected", False)
        multi_face = liveness_payload.get("multi_face", False)

        # ---- hard overrides, same rule as main.py's generate_json() ----
        hard_deny_reasons = []
        if not face_detected:
            hard_deny_reasons.append("no_face_detected")
        if multi_face:
            hard_deny_reasons.append("multiple_faces")
        if anti_spoof_2d_flag:
            hard_deny_reasons.append("2d_spoof_detected_sustained")
        if virtual_camera_detected:
            hard_deny_reasons.append("virtual_camera_detected")

        return {
            "module": "fusion",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "face_detected": face_detected,
            "multi_face_alert": multi_face,
            "signals": {
                "visual_deepfake_probability": deepfake_result.get("deepfake_probability"),
                "identity_similarity": (identity_result or {}).get("similarity_score"),
                "injection_risk_score": injection_risk_score,
                "virtual_camera_detected": virtual_camera_detected,
            },
            "hard_deny_reasons": hard_deny_reasons,
            # None unless a challenge happened to resolve (pass/fail) on
            # this exact frame -- same as main.py's `challenge` variable.
            "challenge_result": challenge,
        }
