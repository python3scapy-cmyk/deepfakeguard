#!/usr/bin/env python3
"""
DeepfakeGuard - Headless Analysis Engine (Faza 0)

Extracts the per-frame analysis logic out of main.py's run() loop into an
importable, camera-free, GUI-free class so the SAME pipeline can be driven
from three places:

  1. backend/app.py  -> Socket.IO 'analysis_frame' events from participant.html
  2. main.py         -> local cv2 camera loop (debug / presenter mode)
  3. /analyze-upload -> offline file analysis (Faza 4)

Key differences from main.py's inline loop:
  - No cv2.VideoCapture, no cv2.imshow, no input(), no cv2.flip
    (participant.html already mirrors its preview; analysis wants the
    raw frame).
  - remote=True sessions do NOT create SecurityMonitor or MicrophoneStream:
    injection timing checks are meaningless for a browser-supplied stream
    (documented policy - injection signal is None and therefore excluded
    from the trust weights, exactly like compute_trust already handles).
  - Heavy models (SigLIP deepfake classifier, XGBoost audio, InsightFace)
    are loaded ONCE at module level and shared across all sessions.
    Per-session state (spoof history, challenge state machine, frame
    counter, session-continuity embedding) lives on the engine instance.
  - Challenge results are BUFFERED (_pending_challenge_result) so a
    challenge that completes between two fusion emissions is attached to
    the next emission instead of being silently lost (main.py only
    reported a challenge if it happened to land on a %30 frame).
  - Identity for remote sessions uses SESSION CONTINUITY: the first good
    single-face frame is enrolled as the session's reference embedding,
    and every ~15th frame is compared against it. This catches "person
    swapped mid-scan / switched to a video" without requiring public
    users to pre-enroll.

The fusion payload shape, WEIGHTS, band thresholds and hard-deny rules are
copied 1:1 from main.py / backend/app.py so backend's ingest mapping and
the dashboard keep working unchanged.
"""
import base64
import hashlib
import os
import json
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np

from vision.face_detector import FaceLandmarkDetector
from vision.liveness_engine import LivenessChallengeEngine
from vision.deepfake_detector import DeepfakeDetector
from audio_module.lip_sync import LipSyncDetector
from audio_module.aasist_detector import AASISTDetector
from audio_module.speech_challenge import SpeechChallengeVerifier


# ─────────────────────────────────────────────────────────────
# Tunables (5 fps browser stream assumed)
# ─────────────────────────────────────────────────────────────
EMIT_EVERY = 5           # emit a fusion payload every N frames (~0.5s at 10fps)
IDENTITY_EVERY = 15      # (legacy, local main.py cadence) - remote path is time-based below
DEEPFAKE_EVERY = 3       # (legacy, local main.py cadence) - remote path is time-based below
# Heavy models are the ONLY reason the pipeline ever feels slow: SigLIP is
# 300-800ms and InsightFace 100-300ms on CPU, and every one of those runs
# blocks a frame that the (millisecond-cheap) liveness path needed. Both
# now run rarely, and NEVER while a challenge is on screen.
DF_INTERVAL_SEC = 3.0    # remote: SigLIP at most once per 3s
ID_INTERVAL_SEC = 6.0    # remote: ArcFace at most once per 6s
NO_FACE_STREAK_DENY = 5  # consecutive face-less processed frames before hard deny
CHALLENGE_COOLDOWN_FRAMES = 15   # ~1.5s at 10fps (was 6s of dead air)
CHALLENGE_TIMEOUT_SEC = 8.0      # keep in sync with LivenessChallengeEngine
SPOOF_WINDOW = 15        # same smoothing window as main.py
SPOOF_MAJORITY = 0.6     # same majority threshold as main.py

# Same weights and bands as main.py / backend/app.py - DO NOT change one
# without changing all three (see WEIGHT_RATIONALE.md).
WEIGHTS = {
    "identity": 0.25,
    "liveness": 0.25,
    "visual_deepfake": 0.20,
    "audio_spoof": 0.20,
    "injection": 0.10,
}


def _has_faces(faces):
    if faces is None:
        return False
    if isinstance(faces, np.ndarray):
        return faces.size > 0
    return len(faces) > 0


def _cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def fuse_scores(scores):
    """Weighted trust fusion - 1:1 with main.py / backend bands. None
    signals are excluded from the weighted average. Module-level so both
    the live engine and offline analyze_file() share one implementation."""
    weighted_sum, total_weight = 0.0, 0.0
    for key, weight in WEIGHTS.items():
        val = scores.get(key)
        if val is not None:
            weighted_sum += val * weight
            total_weight += weight

    if total_weight == 0:
        return {"trust_score": 50.0, "band": "suspicious",
                "lowest_signal": "none", "lowest_score": 0}

    trust_score = weighted_sum / total_weight
    if trust_score >= 80:
        band = "trusted"
    elif trust_score >= 50:
        band = "suspicious"
    else:
        band = "fraud"

    valid = [(k, v) for k, v in scores.items() if v is not None]
    lowest = min(valid, key=lambda kv: kv[1]) if valid else ("none", 0)

    return {
        "trust_score": round(trust_score, 1),
        "band": band,
        "lowest_signal": lowest[0],
        "lowest_score": lowest[1],
    }


# ─────────────────────────────────────────────────────────────
# Shared heavy resources - loaded once per process, used by all
# sessions. Inference calls are serialized with _infer_lock because
# neither SigLIP-on-CPU nor onnxruntime sessions are guaranteed
# thread-safe under Flask-SocketIO's threaded handlers.
# ─────────────────────────────────────────────────────────────
_shared = {}
_shared_lock = threading.Lock()
_infer_lock = threading.Lock()


def _get_shared():
    """Lazy-load the heavy detectors exactly once."""
    with _shared_lock:
        if not _shared:
            print("[ENGINE] Loading shared models (first call only)...")
            _shared["deepfake"] = DeepfakeDetector()      # SigLIP (~370MB, cached after first download)
            _shared["audio"] = AASISTDetector()           # XGBoost ASVspoof checkpoint
            _shared["speech"] = SpeechChallengeVerifier() # ASR loads lazily on first clip
            try:
                from insightface.app import FaceAnalysis
                face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
                face_app.prepare(ctx_id=0, det_size=(320, 320))
                _shared["face_app"] = face_app
                print("[ENGINE] InsightFace loaded (session-continuity identity enabled)")
            except Exception as e:
                print(f"[ENGINE][WARN] InsightFace unavailable ({e}) - "
                      f"identity will report 'insightface_unavailable'.")
                _shared["face_app"] = None
            print("[ENGINE] Shared models ready")
        return _shared


# ─────────────────────────────────────────────────────────────
# Session-continuity identity (remote sessions)
# ─────────────────────────────────────────────────────────────
class SessionIdentity:
    """Identity for public/remote users who have no pre-enrolled face.

    Policy: the first good single-face frame becomes the session's
    reference embedding ("this is whoever started the scan"), and later
    frames are verified against it. A drop in similarity means the person
    in front of the camera changed mid-session (face swap to a different
    identity, someone else sat down, a video was substituted).

    Output dict shape matches security.identity.IdentityMatcher's
    documented contract so the fusion payload is identical either way.
    """

    NOT_ENROLLED_SCORE = 50  # neutral until enrollment happens (unlike the
                             # local pre-enrollment flow, "not yet enrolled"
                             # here just means "first frames haven't arrived",
                             # which shouldn't read as fraud)
    MIN_DET_SCORE = 0.55     # don't enroll from a garbage detection

    def __init__(self, face_app, threshold=0.6):
        self.face_app = face_app
        self.threshold = threshold
        self.embedding = None
        self.enrolled = False
        self._last_score = self.NOT_ENROLLED_SCORE

    def last_score(self):
        return self._last_score

    def try_enroll(self, frame):
        """Attempt to enroll from this frame. Returns True on success."""
        if self.face_app is None or self.enrolled:
            return self.enrolled
        with _infer_lock:
            detected = self.face_app.get(frame)
        if len(detected) != 1:
            return False
        face = detected[0]
        if getattr(face, "det_score", 1.0) < self.MIN_DET_SCORE:
            return False
        self.embedding = face.normed_embedding
        self.enrolled = True
        print("[ENGINE] Session identity enrolled from live frame")
        return True

    def compute_identity_score(self, frame, faces=None):
        start = time.time()

        if self.face_app is None or not self.enrolled:
            result = {
                "identity_score": self.NOT_ENROLLED_SCORE,
                "similarity_score": 0.0,
                "threshold_used": self.threshold,
                "face_detected": _has_faces(faces),
                "multiple_faces_detected": _has_faces(faces) and len(faces) > 1,
                "confidence": "none",
                "latency_ms": round((time.time() - start) * 1000, 1),
                "verdict": "UNKNOWN",
                "reason": ("insightface_unavailable" if self.face_app is None
                           else "session_not_yet_enrolled"),
            }
            self._last_score = result["identity_score"]
            return result

        with _infer_lock:
            detected = self.face_app.get(frame)

        if len(detected) == 0:
            result = {
                "identity_score": 0, "similarity_score": 0.0,
                "threshold_used": self.threshold,
                "face_detected": False, "multiple_faces_detected": False,
                "confidence": "none",
                "latency_ms": round((time.time() - start) * 1000, 1),
                "verdict": "FAKE",
            }
        elif len(detected) > 1:
            result = {
                "identity_score": 0, "similarity_score": 0.0,
                "threshold_used": self.threshold,
                "face_detected": True, "multiple_faces_detected": True,
                "confidence": "none",
                "latency_ms": round((time.time() - start) * 1000, 1),
                "verdict": "FAKE",
            }
        else:
            score = round(_cosine_similarity(detected[0].normed_embedding,
                                             self.embedding), 3)
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


# ─────────────────────────────────────────────────────────────
# The engine
# ─────────────────────────────────────────────────────────────
class AnalysisEngine:
    """One instance per live session (browser participant OR local camera).

    Usage:
        eng = AnalysisEngine(session_id="web_ab12", remote=True)
        fusion = eng.process_frame(frame_bgr)   # dict every EMIT_EVERY frames, else None
        eng.process_audio(pcm_float32_16k)       # optional, 1s chunks
        eng.close()
    """

    def __init__(self, session_id, remote=True, identity_matcher=None):
        shared = _get_shared()
        self._deepfake = shared["deepfake"]
        self._audio = shared["audio"]

        self.session_id = session_id
        self.remote = remote

        # Per-session, per-frame components (these hold cross-frame state
        # and must never be shared between sessions).
        self.vision_detector = FaceLandmarkDetector()
        self.liveness_engine = LivenessChallengeEngine()
        self.lip_sync = LipSyncDetector(window_ms=200)

        # Identity:
        #  - remote sessions -> session-continuity enrollment (SessionIdentity)
        #  - local main.py   -> caller passes its pre-enrolled IdentityMatcher
        if identity_matcher is not None:
            self.identity = identity_matcher
            self._identity_is_session = False
        else:
            self.identity = SessionIdentity(shared["face_app"])
            self._identity_is_session = True

        # Per-session state, copied 1:1 from main.py's __init__/run()
        self.frame_count = 0
        self.spoof_history = deque(maxlen=SPOOF_WINDOW)
        self.df_window = deque(maxlen=30)        # per-session deepfake smoothing
        self.challenge_active = False
        self.challenge_cooldown = 0
        self.active_challenge_type = None

        self._pending_challenge_result = None    # buffered until next emission
        self._last_challenge_passed = None       # None until first challenge resolves
        self._last_deepfake_result = None
        self._last_audio_result = None
        self._last_identity_result = None
        self._last_fusion = None
        self._last_df_time = 0.0                 # time-based heavy-inference gating
        self._last_id_time = 0.0
        self._no_face_streak = 0                 # consecutive processed frames without a face

        # ---- audio (spoken-phrase) challenge - second auth factor ----
        # Issued automatically once the FIRST vision challenge passes,
        # so the demo flow is: vision factor -> voice factor.
        self._speech = _get_shared()["speech"]
        self._face_app = _get_shared().get("face_app")
        self._last_insight_ts = 0.0
        self.audio_challenge = None            # {"phrase", "issued_at", "timeout_sec"}
        self._voice_grace_phrase = None        # phrase kept alive briefly past timeout
        self._voice_grace_until = 0.0
        # Session finalization: once BOTH factors have resolved, the session
        # reaches a final verdict and STOPS re-challenging. Re-running
        # challenges forever after a decision is both bad UX and bad
        # security theatre - a verification either concluded or it didn't.
        self.session_final = False
        self.final_verdict = None
        self._voice_pending_clip = False   # prompt shown, clip not yet verified
        self._audio_challenge_result = None    # last verify() dict (buffered for emission)
        self._audio_challenge_done = False     # one voice factor per session (demo scope)
        self._audio_force_emit = False

        self.last_seen = time.time()
        self._busy = False
        self._busy_lock = threading.Lock()

    # ---------------- audio ----------------
    def process_audio_clip(self, pcm_16k_f32):
        """A recorded voice-challenge clip from the browser. Runs BOTH
        checks on the same audio: phrase verification (liveness: a live
        human said the random phrase) and AASIST spoof scoring (the voice
        is not synthetic/replayed). Result is attached to the next fusion
        emission, which is forced so the participant sees it immediately."""
        self.last_seen = time.time()
        in_grace = (self.audio_challenge is None
                    and self._voice_grace_phrase is not None
                    and time.time() <= self._voice_grace_until)
        if self.audio_challenge is None and not in_grace:
            print(f"[ENGINE][{self.session_id}] audio clip received but no "
                  f"active voice challenge - scoring spoof only")
        elif in_grace:
            phrase = self._voice_grace_phrase
            self._voice_grace_phrase = None
            result = self._speech.verify(pcm_16k_f32, phrase)
            result["phrase"] = phrase
            result["reason"] = (result.get("reason") or "") + " (late clip - grace window)"
            self._audio_challenge_result = result
            self._voice_pending_clip = False
            self._audio_force_emit = True
            print(f"[ENGINE][{self.session_id}] VOICE CHALLENGE (grace) "
                  f"{'PASSED' if result['passed'] else 'FAILED'}: {result.get('reason')}")
        else:
            result = self._speech.verify(pcm_16k_f32,
                                         self.audio_challenge["phrase"])
            result["phrase"] = self.audio_challenge["phrase"]
            self._audio_challenge_result = result
            self._audio_challenge_done = True
            self._voice_pending_clip = False
            self.audio_challenge = None
            self._audio_force_emit = True
            print(f"[ENGINE][{self.session_id}] VOICE CHALLENGE "
                  f"{'PASSED' if result['passed'] else 'FAILED'}: "
                  f"{result.get('reason')}")
        try:
            with _infer_lock:
                self._last_audio_result = self._audio.ensemble_score(pcm_16k_f32)
        except Exception as e:
            print(f"[ENGINE][WARN] audio spoof scoring failed: {e}")

    def process_audio(self, pcm_float32):
        """Feed ~1s of 16kHz mono float32 PCM from the browser (Faza 1/P1).
        Safe to never call - audio signal then stays neutral (50)."""
        self.last_seen = time.time()
        try:
            with _infer_lock:
                self._last_audio_result = self._audio.ensemble_score(pcm_float32)
        except Exception as e:
            print(f"[ENGINE][WARN] audio scoring failed: {e}")

    # ---------------- fusion math (1:1 with main.py) ----------------
    def compute_trust_score(self, scores):
        return fuse_scores(scores)

    # ---------------- main entry point ----------------
    def process_frame(self, frame_bgr, ear=None, yaw_ratio=None):
        """Analyze one frame. Returns a fusion payload dict every
        EMIT_EVERY frames, otherwise None.

        Frame-drop guard: if a previous frame is still being processed
        (Socket.IO handler overlap), the new frame is discarded instead
        of queueing up and snowballing latency.
        """
        with self._busy_lock:
            if self._busy:
                return None
            self._busy = True
        try:
            return self._process_frame_inner(frame_bgr, ear=ear, yaw_ratio=yaw_ratio)
        finally:
            with self._busy_lock:
                self._busy = False

    def _process_frame_inner(self, frame, ear=None, yaw_ratio=None):
        self.frame_count += 1
        now = time.time()
        self.last_seen = now

        # ---- vision + liveness (same order as main.py) ----
        faces, vision_payload = self.vision_detector.process_frame(frame)

        # SCRFD fallback + REAL head yaw. Haar (frontal) collapses on
        # pitched-down faces (users look DOWN to read the prompt panel
        # under the video - confirmed frame-by-frame on a user recording)
        # and on turned faces. InsightFace is already loaded for identity;
        # its detector is robust to pitch/turn AND its 3D pose gives true
        # yaw in degrees, which the liveness engine treats as definitive
        # turn evidence. Run it only when it matters (turn challenge
        # active, or Haar found nothing), throttled to ~6/sec.
        insight_yaw = None
        turn_active = (self.challenge_active
                       and self.active_challenge_type in ("turn_left", "turn_right"))
        now0 = time.time()
        if (self._face_app is not None
                and (turn_active or not _has_faces(faces))
                and now0 - self._last_insight_ts >= 0.30):
            self._last_insight_ts = now0
            try:
                with _infer_lock:
                    dets = self._face_app.get(frame)
            except Exception:
                dets = []
            if len(dets) > 0:
                f0 = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
                x1, y1, x2, y2 = [int(v) for v in f0.bbox]
                if not _has_faces(faces):
                    faces = [(max(0, x1), max(0, y1), max(1, x2 - x1), max(1, y2 - y1))]
                pose = getattr(f0, "pose", None)
                if pose is not None and len(pose) >= 2:
                    insight_yaw = float(pose[1])

        _, liveness_payload = self.liveness_engine.process_frame(
            frame, faces, true_yaw=insight_yaw, ear=ear, mesh_yaw=yaw_ratio)

        if liveness_payload.get("face_detected"):
            self._no_face_streak = 0
        else:
            self._no_face_streak += 1

        # ---- session-continuity enrollment (remote only) ----
        if (self._identity_is_session and not self.identity.enrolled
                and liveness_payload.get("face_detected")
                and not liveness_payload.get("multi_face", False)):
            self.identity.try_enroll(frame)

        # ---- heavy inference gating ----
        # SigLIP takes 300-800ms on CPU. Run inline every Nth frame and the
        # busy-drop guard throws away most incoming frames, starving the
        # liveness engine down to ~2 effective fps - head turns and blinks
        # become physically undetectable and every challenge times out
        # ("0ms response"). Two rules fix this:
        #   1. While a challenge is ACTIVE, heavy inference pauses entirely:
        #      every frame goes to the (fast) vision/liveness path.
        #   2. Otherwise, deepfake/identity run on WALL-CLOCK intervals, not
        #      frame counts, so they can never consume most of the stream.
        heavy_allowed = not self.challenge_active

        if heavy_allowed and (now - self._last_df_time) >= DF_INTERVAL_SEC:
            self._last_df_time = now
            try:
                with _infer_lock:
                    df = self._deepfake.analyze_single_frame(frame)
                self._last_deepfake_result = df
                if df.get("face_found", True) and df.get("deepfake_probability") is not None:
                    self.df_window.append(float(df["deepfake_probability"]))
            except Exception as e:
                print(f"[ENGINE][WARN] deepfake scoring failed: {e}")

        # ---- lip-sync proxy (silence when no browser audio yet) ----
        mouth_crop = self.liveness_engine.get_mouth_crop(frame, faces)
        if mouth_crop is not None:
            self.lip_sync.update_mouth_openness(mouth_crop)
        self.lip_sync.update_audio(np.zeros(1600, dtype=np.float32))
        sync_result = self.lip_sync.compute_sync_score()

        # ---- identity, time-based ----
        identity_result = None
        if heavy_allowed and (now - self._last_id_time) >= ID_INTERVAL_SEC:
            self._last_id_time = now
            identity_result = self.identity.compute_identity_score(frame, faces)
            self._last_identity_result = identity_result

        # ---- challenge state machine (from main.py's run()) ----
        force_emit = False
        challenge = liveness_payload.get("challenge_result")
        if challenge:
            self._pending_challenge_result = challenge
            self._last_challenge_passed = bool(challenge.get("challenge_passed", False))
            self.challenge_active = False
            self.active_challenge_type = None
            self.challenge_cooldown = self.frame_count + CHALLENGE_COOLDOWN_FRAMES
            self.liveness_engine.reset()
            force_emit = True   # participant HUD should see PASS/FAIL immediately
        elif (not self.session_final
                and not self.challenge_active
                and self.audio_challenge is None          # voice factor has the floor
                and not self._voice_pending_clip
                and self.frame_count > self.challenge_cooldown
                and liveness_payload.get("face_detected")
                and liveness_payload.get("state") == "IDLE"):
            new_challenge = self.liveness_engine.issue_challenge()
            self.challenge_active = True
            self.active_challenge_type = new_challenge
            force_emit = True   # announce the prompt without waiting for the next %EMIT frame
            print(f"[ENGINE][{self.session_id}] NEW CHALLENGE: {new_challenge}")

        # ---- voice challenge lifecycle ----
        # Voice is an INDEPENDENT second factor: it fires once the first
        # vision challenge RESOLVES (pass or fail), not only on success -
        # otherwise a user stuck on turn challenges would never reach the
        # voice factor and the audio panel would sit "not connected"
        # forever (exactly the observed dashboard state).
        if (not self.session_final
                and self._last_challenge_passed is not None
                and not self._audio_challenge_done
                and self.audio_challenge is None and not self.challenge_active):
            self.audio_challenge = {
                "phrase": self._speech.new_phrase(),
                "issued_at": now,
                "timeout_sec": 16.0,
            }
            self._voice_pending_clip = True
            force_emit = True
            print(f"[ENGINE][{self.session_id}] NEW VOICE CHALLENGE: "
                  f"say '{self.audio_challenge['phrase']}'")
        if (self.audio_challenge is not None
                and now - self.audio_challenge["issued_at"] > self.audio_challenge["timeout_sec"]):
            self._audio_challenge_result = {
                "passed": False, "mode": "timeout",
                "phrase": self.audio_challenge["phrase"],
                "expected_phrase": self.audio_challenge["phrase"],
                "transcript": None, "match_ratio": 0.0,
                "reason": "no voice response within the time limit",
            }
            self._audio_challenge_done = True
            self._voice_pending_clip = False
            self._voice_grace_phrase = self.audio_challenge["phrase"]
            self._voice_grace_until = now + 20.0   # late-clip grace window
            self.audio_challenge = None
            force_emit = True
        if self._audio_force_emit:
            force_emit = True
            self._audio_force_emit = False

        active_audio_challenge = None
        if self.audio_challenge is not None:
            active_audio_challenge = {
                "phrase": self.audio_challenge["phrase"],
                "remaining_sec": max(0, int(self.audio_challenge["timeout_sec"]
                                            - (now - self.audio_challenge["issued_at"]))),
            }

        active_challenge = None
        if self.challenge_active:
            elapsed = time.time() - self.liveness_engine.challenge_start_time
            active_challenge = {
                "challenge_type": (liveness_payload.get("challenge_type")
                                   or self.active_challenge_type),
                "remaining_sec": max(0, int(CHALLENGE_TIMEOUT_SEC - elapsed)),
            }

        # ---- spoof smoothing (1:1 with main.py) ----
        spoof_analysis = vision_payload.get("spoof_analysis") or {}
        raw_spoof_flag = (spoof_analysis.get("is_spoof", False)
                          or liveness_payload.get("is_static", False))
        self.spoof_history.append(raw_spoof_flag)
        spoof_ratio = sum(self.spoof_history) / len(self.spoof_history)
        anti_spoof_2d_flag = (spoof_ratio >= SPOOF_MAJORITY
                              and len(self.spoof_history) >= 12)

        # ---- signal scores (same semantics as main.py) ----
        identity_trust = (identity_result["identity_score"]
                          if identity_result else self.identity.last_score())

        visual_df_score = int(np.mean(self.df_window) * 100) if self.df_window else 50

        # ABSENT signal != NEUTRAL evidence. A signal that never arrived
        # (no browser audio pipeline; no challenge attempted yet) is
        # excluded from the weighted average via None - the same treatment
        # injection already gets. The old flat 50 permanently dragged real
        # users' trust below the 80 'trusted' threshold no matter how well
        # every ACTIVE signal scored.
        if self._last_audio_result is not None:
            audio_trust = 100 - float(self._last_audio_result.get("spoof_probability", 0.5)) * 100
        else:
            audio_trust = None  # no audio data -> excluded from weights
        # A failed voice challenge is strong negative evidence regardless of
        # what AASIST thought of the raw signal.
        if (self._audio_challenge_result is not None
                and self._audio_challenge_result.get("passed") is False):
            audio_trust = min(audio_trust, 25) if audio_trust is not None else 25

        _df_now = float(np.mean(self.df_window)) if self.df_window else None
        _corroborated = (anti_spoof_2d_flag and _df_now is not None and _df_now >= 0.55)
        if _corroborated:
            liveness_trust = 20          # heuristic AND model agree -> spoof
        elif anti_spoof_2d_flag:
            liveness_trust = 45          # heuristic alone -> suspicion, not a verdict
        elif self._last_challenge_passed is True:
            liveness_trust = 100
        elif self._last_challenge_passed is False:
            liveness_trust = 30   # an attempted-and-failed challenge is evidence
        else:
            liveness_trust = None  # no challenge resolved yet -> excluded

        scores = {
            "identity": identity_trust,
            "liveness": liveness_trust,
            "visual_deepfake": 100 - visual_df_score,
            "audio_spoof": audio_trust,
            # remote browser stream: injection timing check is not
            # applicable -> None -> excluded from the weighted average
            # (compute_trust_score skips None, and backend's mapping is
            # guarded by `if injection_risk_score is not None`).
            "injection": None,
        }
        trust_data = self.compute_trust_score(scores)

        # ---- finalize the session once both factors have resolved ----
        if (not self.session_final
                and self._last_challenge_passed is not None
                and self._audio_challenge_done
                and self.audio_challenge is None):
            self.session_final = True
            force_emit = True
            print(f"[ENGINE][{self.session_id}] SESSION FINAL - "
                  f"trust={trust_data['trust_score']} band={trust_data['band']}")

        # ---- emit fusion payload every EMIT_EVERY frames, or immediately
        # when a challenge is issued/resolved ----
        if not force_emit and self.frame_count % EMIT_EVERY != 0:
            return None

        if active_audio_challenge is not None:
            print(f"[ENGINE][{self.session_id}] emit carries voice prompt "
                  f"'{active_audio_challenge['phrase']}' "
                  f"({active_audio_challenge['remaining_sec']}s left)")

        fusion = self._generate_fusion(
            liveness_payload, spoof_analysis, anti_spoof_2d_flag,
            trust_data, sync_result, identity_result, active_challenge,
            active_audio_challenge,
        )
        self._pending_challenge_result = None  # consumed by this emission
        self._last_fusion = fusion
        return fusion

    # ---------------- payload builder (1:1 field parity with main.py) ----------------
    def _generate_fusion(self, liveness_payload, spoof_analysis, anti_spoof_2d_flag,
                         trust_data, sync_result, identity_result, active_challenge,
                         active_audio_challenge=None):
        challenge = self._pending_challenge_result
        face_detected = liveness_payload.get("face_detected", False)
        multi_face = liveness_payload.get("multi_face", False)

        hard_deny_reasons = []
        # A single dropped detection (motion blur, JPEG artifact, head at
        # frame edge) must not flip the verdict to DENIED - require a
        # sustained streak of face-less frames, mirroring how the 2D spoof
        # flag already needs a majority window.
        if not face_detected and self._no_face_streak >= NO_FACE_STREAK_DENY:
            hard_deny_reasons.append("no_face_detected")
        if multi_face:
            hard_deny_reasons.append("multiple_faces")
        df_mean_now = float(np.mean(self.df_window)) if self.df_window else None
        model_agrees_fake = (df_mean_now is not None and df_mean_now >= 0.55)
        if anti_spoof_2d_flag and model_agrees_fake:
            hard_deny_reasons.append("2d_spoof_detected_sustained")
        # Session-continuity break = the face changed mid-scan. Treated as
        # a hard deny only on a confident mismatch (score computed AND low).
        ident = identity_result or self._last_identity_result
        if (self._identity_is_session and self.identity.enrolled and ident
                and ident.get("face_detected")
                and not ident.get("multiple_faces_detected")
                and ident.get("similarity_score", 1.0) < 0.35):
            hard_deny_reasons.append("identity_continuity_broken")

        if hard_deny_reasons:
            verdict = "ACCESS DENIED"
        else:
            verdict = ("ACCESS GRANTED"
                       if trust_data["band"] in ("trusted", "suspicious")
                       else "ACCESS DENIED")

        df = self._last_deepfake_result
        visual_df_prob = (round(float(np.mean(self.df_window)), 4)
                          if self.df_window else None)
        audio_prob = (self._last_audio_result.get("spoof_probability")
                      if self._last_audio_result else None)

        return {
            "module": "fusion",
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "face_detected": face_detected,
            "multi_face_alert": multi_face,
            "trust_score": trust_data["trust_score"],
            "trust_band": trust_data["band"],
            "lowest_signal": trust_data["lowest_signal"],
            "signals": {
                "liveness_challenge_passed": bool(challenge and challenge.get("challenge_passed")),
                "visual_deepfake_probability": visual_df_prob,
                "audio_spoof_probability": audio_prob,
                "lip_sync_score": sync_result.get("lip_sync_score") if sync_result else None,
                "anti_spoof_2d_raw": spoof_analysis.get("is_spoof", False),
                "anti_spoof_2d_sustained": anti_spoof_2d_flag,
                "identity_similarity": (ident.get("similarity_score") if ident else None),
                "injection_risk_score": None,          # N/A for remote streams
                "virtual_camera_detected": None,       # N/A for remote streams
            },
            "hard_deny_reasons": hard_deny_reasons,
            "verdict": verdict,
            "challenge_result": challenge,
            # Extra (not in main.py): lets participant.html render the
            # prompt + countdown. backend/app.py ignores unknown keys.
            "active_challenge": active_challenge,
            # Voice (second) factor - Faza Audio:
            "active_audio_challenge": active_audio_challenge,
            "audio_challenge_result": self._audio_challenge_result,
            "factors": {
                "vision_challenge_passed": self._last_challenge_passed,
                "audio_challenge_passed": (self._audio_challenge_result.get("passed")
                                           if self._audio_challenge_result else None),
            },
            "session_final": self.session_final,
            "source": "engine_remote" if self.remote else "engine_local",
        }

    def last_fusion(self):
        return self._last_fusion

    def to_json(self, fusion_dict):
        """For callers that need the string form (main.py's /fusion POST)."""
        return json.dumps(fusion_dict, indent=2)

    def close(self):
        try:
            self.liveness_engine.close()
        except Exception:
            pass
        try:
            self.vision_detector.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# Tiny session manager (backend will use this in Faza 1)
# ─────────────────────────────────────────────────────────────
def warm_up():
    """Preload every shared model (SigLIP, XGBoost audio, InsightFace, and
    the Whisper ASR) at backend startup instead of on the first live frame/
    clip - first-use latency killed the first voice challenge of a session
    (75MB whisper download/load > challenge timeout)."""
    shared = _get_shared()
    try:
        shared["speech"]._ensure_loaded()
    except Exception as e:
        print(f"[ENGINE][WARN] speech warm-up failed: {e}")
    print("[ENGINE] Warm-up complete - all shared models resident")


ENGINES = {}
_engines_lock = threading.Lock()
MAX_ACTIVE_SESSIONS = 3
STALE_AFTER_SEC = 60


def get_engine(session_id, remote=True):
    """Fetch-or-create the engine for a session. Returns None if the
    concurrency cap is reached (caller should tell the user to wait)."""
    with _engines_lock:
        eng = ENGINES.get(session_id)
        if eng is not None:
            return eng
        drop_stale_locked()
        if len(ENGINES) >= MAX_ACTIVE_SESSIONS:
            return None
        eng = AnalysisEngine(session_id=session_id, remote=remote)
        ENGINES[session_id] = eng
        print(f"[ENGINE] Session started: {session_id} "
              f"({len(ENGINES)}/{MAX_ACTIVE_SESSIONS} active)")
        return eng


def drop_stale_locked():
    now = time.time()
    for sid in [s for s, e in ENGINES.items() if now - e.last_seen > STALE_AFTER_SEC]:
        try:
            ENGINES[sid].close()
        except Exception:
            pass
        del ENGINES[sid]
        print(f"[ENGINE] Stale session dropped: {sid}")


def drop_session(session_id):
    with _engines_lock:
        eng = ENGINES.pop(session_id, None)
    if eng is not None:
        eng.close()
        print(f"[ENGINE] Session closed: {session_id}")


# ─────────────────────────────────────────────────────────────
# Faza 3: offline full-pipeline analysis for /analyze-upload
# ─────────────────────────────────────────────────────────────
# Deterministic re-analysis: the same uploaded file always yields the same
# verdict, and a repeat upload is instant. Bounded so a long session can't
# grow it without limit (the teammates' version was unbounded).
_upload_cache = {}
_UPLOAD_CACHE_MAX = 32


def _file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_get(fhash):
    return _upload_cache.get(fhash)


def _cache_put(fhash, result):
    if len(_upload_cache) >= _UPLOAD_CACHE_MAX:
        _upload_cache.pop(next(iter(_upload_cache)))
    _upload_cache[fhash] = result
    return result


# Audio-only verdict thresholds. The XGBoost model is calibrated so 0.5 is
# its EER decision point, but on real TTS/voice-clone samples the team
# measured spoof probabilities clustering at 0.25-0.45 - i.e. genuinely
# synthetic audio often lands BELOW the calibrated point. A 0.30 flag
# threshold is therefore empirical, not arbitrary; the band between 0.20
# and 0.30 is reported as SUSPICIOUS rather than forced into a hard verdict.
AUDIO_FAKE_THRESHOLD = 0.30
AUDIO_SUSPICIOUS_THRESHOLD = 0.20


def analyze_audio_file(path, max_seconds=8):
    """Offline analysis for an audio-only upload (wav/mp3/m4a/flac/ogg/aac).

    No visual signals exist, so visual_deepfake, liveness, identity and
    injection are all excluded from the fusion (None), exactly as
    analyze_file() already does for the signals that don't apply to it."""
    fhash = _file_hash(path)
    cached = _cache_get(fhash)
    if cached is not None:
        print("[ENGINE] upload cache hit (audio)")
        return cached

    shared = _get_shared()
    try:
        import librosa
        y, _sr = librosa.load(path, sr=16000, mono=True, duration=max_seconds)
    except Exception as e:
        return {"error": f"could not decode audio: {e}. "
                         f"(mp3/m4a need ffmpeg installed)"}

    if y is None or len(y) < 1600:      # <100ms
        return {"error": "audio too short or unreadable"}

    y = y.astype(np.float32)
    try:
        with _infer_lock:
            audio_result = shared["audio"].ensemble_score(y)
    except Exception as e:
        return {"error": f"audio scoring failed: {e}"}

    audio_prob = float(audio_result.get("spoof_probability"))
    duration_s = round(len(y) / 16000.0, 1)

    trust_data = fuse_scores({
        "identity": None,
        "liveness": None,
        "visual_deepfake": None,
        "audio_spoof": 100 - audio_prob * 100,
        "injection": None,
    })

    if audio_prob >= AUDIO_FAKE_THRESHOLD:
        verdict = "FAKE"
        confidence_pct = round(audio_prob * 100, 1)
    elif audio_prob >= AUDIO_SUSPICIOUS_THRESHOLD:
        verdict = "SUSPICIOUS"
        confidence_pct = round(audio_prob * 100, 1)
    else:
        verdict = "REAL"
        confidence_pct = round((1 - audio_prob) * 100, 1)

    result = {
        "verdict": verdict,
        "confidence_pct": confidence_pct,
        "media_kind": "audio",
        "duration_sec": duration_s,
        "frames_analyzed": 0,
        "frames": [],
        "model_backend": audio_result.get("model", "unknown"),
        "full_pipeline": {
            "trust_score": trust_data["trust_score"],
            "trust_band": trust_data["band"],
            "lowest_signal": trust_data["lowest_signal"],
            "hard_deny_reasons": [],
            "signals": {
                "audio_spoof_probability": round(audio_prob, 4),
                "visual_deepfake_probability": None,
                "identity_continuity_min": None,
                "identity_continuity_mean": None,
                "spoof_frame_ratio": None,
                "liveness": "not_applicable_audio_only",
                "injection": "not_applicable_audio_only",
            },
            "audio_note": "ok ({}, {}s analysed)".format(
                audio_result.get("model", "unknown"), duration_s),
        },
    }
    return _cache_put(fhash, result)


def _thumbnail_b64(frame, max_dim=160):
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _extract_audio_16k(video_path, max_seconds=8):
    """Pull the audio track out of a video with ffmpeg and return
    (float32 mono 16kHz array, note). note explains why audio is missing
    when it is - the dashboard shows it instead of silently skipping."""
    if shutil.which("ffmpeg") is None:
        return None, "ffmpeg_not_installed"
    wav_path = None
    try:
        fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="dfg_audio_")
        os.close(fd)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
             "-ar", "16000", "-t", str(max_seconds), wav_path],
            capture_output=True, timeout=60,
        )
        if proc.returncode != 0:
            return None, "no_audio_track"
        import librosa
        y, _sr = librosa.load(wav_path, sr=16000, mono=True)
        if y is None or len(y) < 1600:  # <100ms of audio is useless
            return None, "audio_too_short"
        return y.astype(np.float32), "ok"
    except Exception as e:
        return None, f"audio_extract_failed: {e}"
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass


def analyze_file(path, is_video, max_frames=20):
    """Run the FULL pipeline on an uploaded photo/video:
    visual deepfake (per sampled frame) + 2D spoof heuristics + identity
    continuity across frames + audio spoof (AASIST on the extracted
    track), fused with the SAME weights/bands as the live path.

    Liveness challenges and injection checks are not applicable offline,
    so both are None and excluded from the weighted average - identical
    semantics to how remote live sessions exclude injection.

    Returns a dict that keeps the legacy /analyze-upload response fields
    (verdict, confidence_pct, mean_deepfake_probability, frames_analyzed,
    model_backend, frames[]) and adds a "full_pipeline" block."""
    fhash = _file_hash(path)
    cached = _cache_get(fhash)
    if cached is not None:
        print("[ENGINE] upload cache hit (image/video)")
        return cached

    shared = _get_shared()
    frame_results = []
    spoof_flags = []
    embeddings = []

    spoof_detector = FaceLandmarkDetector()  # fresh per file: holds state
    try:
        # ---- collect sampled frames ----
        frames = []
        if is_video:
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            sample_indices = None
            if total > 0:
                step = max(1, total // max_frames)
                sample_indices = set(range(0, total, step))
            idx = 0
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if sample_indices is None or idx in sample_indices:
                    frames.append(frame)
                idx += 1
            cap.release()
        else:
            frame = cv2.imread(path)
            if frame is None:
                return {"error": "could not decode image"}
            frames.append(frame)

        if not frames:
            return {"error": "no frames could be read from file"}

        # ---- per-frame: deepfake + spoof heuristics + embedding ----
        for frame in frames:
            with _infer_lock:
                df = shared["deepfake"].analyze_single_frame(frame)
            df["thumbnail"] = _thumbnail_b64(frame)
            frame_results.append(df)

            faces, vision_payload = spoof_detector.process_frame(frame)
            sa = (vision_payload or {}).get("spoof_analysis") or {}
            if _has_faces(faces):
                spoof_flags.append(bool(sa.get("is_spoof", False)))

            if shared["face_app"] is not None:
                with _infer_lock:
                    detected = shared["face_app"].get(frame)
                if len(detected) == 1:
                    embeddings.append(detected[0].normed_embedding)
    finally:
        try:
            spoof_detector.close()
        except Exception:
            pass

    probs = [r["deepfake_probability"] for r in frame_results
             if r.get("face_found") and r.get("deepfake_probability") is not None]
    if not probs:
        return {"verdict": "NO_FACE", "frames": frame_results,
                "frames_analyzed": len(frame_results),
                "full_pipeline": {"note": "no face found in any sampled frame"}}

    mean_prob = float(np.mean(probs))

    # ---- audio (video only) ----
    audio_prob, audio_note = None, ("image_has_no_audio" if not is_video else None)
    if is_video:
        pcm, audio_note = _extract_audio_16k(path)
        if pcm is not None:
            try:
                with _infer_lock:
                    audio_result = shared["audio"].ensemble_score(pcm)
                audio_prob = float(audio_result.get("spoof_probability"))
                audio_note = "ok ({})".format(audio_result.get("model", "unknown"))
            except Exception as e:
                audio_note = f"audio_scoring_failed: {e}"

    # ---- identity continuity across frames ----
    continuity_min = continuity_mean = None
    if len(embeddings) >= 2:
        sims = [_cosine_similarity(embeddings[0], e) for e in embeddings[1:]]
        continuity_min = round(min(sims), 3)
        continuity_mean = round(float(np.mean(sims)), 3)

    spoof_ratio = (sum(spoof_flags) / len(spoof_flags)) if spoof_flags else 0.0

    # ---- fusion (same weights/bands as live) ----
    scores = {
        "identity": (continuity_mean * 100 if continuity_mean is not None else None),
        "liveness": None,                      # challenges impossible offline
        "visual_deepfake": 100 - mean_prob * 100,
        "audio_spoof": (100 - audio_prob * 100) if audio_prob is not None else None,
        "injection": None,                     # not applicable offline
    }
    trust_data = fuse_scores(scores)

    hard_deny_reasons = []
    if spoof_ratio >= SPOOF_MAJORITY and len(spoof_flags) >= 3:
        hard_deny_reasons.append("2d_spoof_detected_sustained")
    if continuity_min is not None and continuity_min < 0.35:
        hard_deny_reasons.append("identity_continuity_broken")

    # Legacy verdict stays deepfake-probability-based for UI continuity,
    # but hard-deny or a fraud band forces FAKE regardless.
    verdict = "FAKE" if (mean_prob > 0.5 or hard_deny_reasons
                         or trust_data["band"] == "fraud") else "REAL"
    confidence_pct = round((mean_prob if verdict == "FAKE" else 1 - mean_prob) * 100, 1)

    return _cache_put(fhash, {
        "verdict": verdict,
        "confidence_pct": confidence_pct,
        "media_kind": "video" if is_video else "image",
        "mean_deepfake_probability": round(mean_prob, 4),
        "frames_analyzed": len(frame_results),
        "model_backend": getattr(shared["deepfake"], "backend", None) or "mock",
        "frames": frame_results,
        "full_pipeline": {
            "trust_score": trust_data["trust_score"],
            "trust_band": trust_data["band"],
            "lowest_signal": trust_data["lowest_signal"],
            "hard_deny_reasons": hard_deny_reasons,
            "signals": {
                "visual_deepfake_probability": round(mean_prob, 4),
                "audio_spoof_probability": audio_prob,
                "identity_continuity_min": continuity_min,
                "identity_continuity_mean": continuity_mean,
                "spoof_frame_ratio": round(spoof_ratio, 3),
                "liveness": "not_applicable_offline",
                "injection": "not_applicable_offline",
            },
            "audio_note": audio_note,
        },
    })
