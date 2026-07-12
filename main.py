#!/usr/bin/env python3
"""
DeepfakeGuard - Week 3 Fusion Engine

Single entry point that wires together:
  - vision.face_detector      (face detection + 2D anti-spoof heuristics)
  - vision.liveness_engine    (blink / head-turn challenge state machine)
  - vision.deepfake_detector  (cascaded visual deepfake classifier)
  - audio_module.lip_sync (mouth/audio correlation proxy)
  - audio_module.aasist_detector (AASIST+XGBoost audio spoof ensemble)
  - security.security_module  (virtual camera / frame-injection detection)
  - security.identity         (ArcFace 1:1 identity matching)

and combines all six signals into one live trust score that mirrors the
backend's fusion contract exactly (same weights, same bands).

Run with:  python3 main.py
"""
import json
import os
import platform
import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np
import argparse
import requests

from vision.face_detector import FaceLandmarkDetector
from vision.liveness_engine import LivenessChallengeEngine
from vision.deepfake_detector import DeepfakeDetector
from audio_module.lip_sync import LipSyncDetector
from audio_module.aasist_detector import AASISTDetector
from security.identity import IdentityMatcher
from security.security_module import get_security_signal_from_deltas

try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except ImportError:
    _HAS_SOUNDDEVICE = False


def _has_faces(faces):
    if faces is None:
        return False
    if isinstance(faces, np.ndarray):
        return faces.size > 0
    return len(faces) > 0


class MicrophoneStream:
    """Small ring-buffer microphone reader.

    Runs a sounddevice InputStream in a background callback and keeps the
    last ~2 seconds of audio available via get_recent_seconds(). If
    sounddevice isn't installed or no input device is available, this
    degrades to returning silence (zeros) rather than crashing the app --
    callers should still treat scores derived from silence with the same
    "insufficient_data" handling they already have.
    """

    def __init__(self, sample_rate=16000, max_seconds=2.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.lock = threading.Lock()
        self.stream = None
        self.available = False

        if not _HAS_SOUNDDEVICE:
            print("[WARN] sounddevice not installed - audio pipeline will use silence. "
                  "Run: pip install sounddevice --break-system-packages")
            return

        try:
            self.stream = sd.InputStream(
                samplerate=sample_rate, channels=1, dtype="float32",
                callback=self._callback,
            )
            self.stream.start()
            self.available = True
        except Exception as e:
            print(f"[WARN] Could not open microphone ({e}) - audio pipeline will use silence.")
            self.stream = None
            self.available = False

    def _callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0].copy()
        with self.lock:
            n = len(chunk)
            if n >= self.max_samples:
                self.buffer = chunk[-self.max_samples:]
            else:
                self.buffer = np.concatenate([self.buffer[n:], chunk])

    def get_recent(self, num_samples):
        with self.lock:
            if num_samples >= self.max_samples:
                return self.buffer.copy()
            return self.buffer[-num_samples:].copy()

    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass


class AsyncWorker:
    """Runs one slow, blocking scoring function (SigLIP deepfake inference,
    ArcFace identity matching, etc.) on its own background thread.

    This is the fix for the freezing/donma problem: score_frame() and
    compute_identity_score() were previously called inline in the 30fps
    camera loop, so every 300-800ms model call stalled the whole window.
    Now the main loop only ever does two cheap, non-blocking things:
      - submit(*args)   -- hand off the newest frame (never waits)
      - get_latest()    -- read whatever the worker last finished (never waits)

    If the worker is still busy on an older frame when a newer one arrives,
    the newer one simply overwrites the pending job -- frames are dropped
    rather than queued, so the worker is always working on the most recent
    input instead of catching up on stale ones.
    """

    def __init__(self, fn, name="worker"):
        self.fn = fn
        self.name = name
        self.lock = threading.Lock()
        self.latest_result = None
        self._pending = None
        self._has_pending = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, *args, **kwargs):
        with self.lock:
            self._pending = (args, kwargs)
            self._has_pending = True

    def _run(self):
        while not self._stop:
            with self.lock:
                if self._has_pending:
                    job = self._pending
                    self._has_pending = False
                else:
                    job = None
            if job is None:
                time.sleep(0.005)
                continue
            args, kwargs = job
            try:
                result = self.fn(*args, **kwargs)
            except Exception as e:
                print(f"[WARN] {self.name} inference failed: {e}")
                result = None
            with self.lock:
                self.latest_result = result

    def get_latest(self):
        with self.lock:
            return self.latest_result

    def close(self):
        self._stop = True


class BackendSync:
    """Runs the backend POST /fusion and GET /scores calls on their own
    background thread instead of inline in the main camera loop.

    Both used to run synchronously in run(), each with timeout=1.0 -- if
    the Flask backend was ever slow to answer (busy loading its own model,
    under load, etc.) that was up to a full second of frozen video PER
    CALL, happening every 15-30 frames. This was a second, independent
    cause of the freezing alongside the SigLIP/ArcFace inference. The main
    loop now only ever calls submit_fusion() (hands off the JSON string,
    never blocks) and get_latest_trust() (reads whatever the last GET
    returned, never blocks) -- all actual HTTP happens here, on its own
    cooldown, off the render path entirely.
    """

    def __init__(self, backend_url, no_backend=False, poll_interval=0.5):
        self.backend_url = backend_url
        self.no_backend = no_backend
        self.poll_interval = poll_interval
        self.lock = threading.Lock()
        self._pending_fusion = None
        self._latest_trust = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_fusion(self, fusion_json_str):
        with self.lock:
            self._pending_fusion = fusion_json_str

    def get_latest_trust(self):
        with self.lock:
            return self._latest_trust

    def _run(self):
        while not self._stop:
            if not self.no_backend:
                with self.lock:
                    payload = self._pending_fusion
                    self._pending_fusion = None
                if payload is not None:
                    try:
                        resp = requests.post(
                            self.backend_url + "/fusion", data=payload,
                            headers={"Content-Type": "application/json"}, timeout=1.0,
                        )
                        if resp.status_code != 200:
                            print(f"[WARN] Backend returned {resp.status_code}: {resp.text[:120]}")
                    except requests.exceptions.ConnectionError:
                        pass  # backend not running -- silent
                    except Exception as e:
                        print(f"[WARN] Backend post failed: {e}")

                try:
                    resp = requests.get(self.backend_url + "/scores", timeout=1.0)
                    if resp.status_code == 200:
                        with self.lock:
                            self._latest_trust = resp.json()
                except Exception:
                    pass
            for _ in range(int(self.poll_interval * 10)):
                if self._stop:
                    return
                time.sleep(0.1)

    def close(self):
        self._stop = True


class SecurityMonitor:
    """Runs the injection/virtual-camera check on a background thread with
    a cooldown, instead of inline in the main camera loop.

    IMPORTANT: this used to open its OWN cv2.VideoCapture(camera_index)
    every cooldown cycle to measure frame-timing variance -- and that was
    the cause of the garbled/static video seen on Windows. Two concurrent
    VideoCapture opens on the same physical device (this one, usually via
    the default backend, plus the main loop's, via DirectShow) is a
    textbook way to corrupt frames; the two backends fight the driver for
    the same buffer. Once the main loop stopped blocking on slow model
    calls (see AsyncWorker), it called cap.read() far more continuously,
    which made the two captures overlap far more often and made the
    corruption much more visible than before.

    Fix: this class no longer opens a capture at all. The main loop calls
    feed_frame() once per iteration (just records time.time() -- no I/O),
    and the background thread computes the exact same timing-anomaly
    signal from those shared timestamps via
    security_module.get_security_signal_from_deltas(). There is now
    exactly one VideoCapture on the device, ever.
    """

    def __init__(self, camera_index=0, session_id="session", cooldown_sec=8.0, window_size=50):
        self.camera_index = camera_index
        self.session_id = session_id
        self.cooldown_sec = cooldown_sec
        self.lock = threading.Lock()
        self.latest = None
        self._stop = False
        self.frame_times = deque(maxlen=window_size + 1)
        self.frame_times_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed_frame(self):
        """Call once per main-loop iteration, right after a successful
        cap.read(). Cheap -- just appends a timestamp, no camera access."""
        with self.frame_times_lock:
            self.frame_times.append(time.time())

    def _run(self):
        while not self._stop:
            try:
                with self.frame_times_lock:
                    times = list(self.frame_times)
                deltas = [t2 - t1 for t1, t2 in zip(times, times[1:])]
                result = get_security_signal_from_deltas(deltas, self.camera_index, self.session_id)
                with self.lock:
                    self.latest = result
            except Exception as e:
                print(f"[WARN] Security/injection check failed: {e}")
            for _ in range(int(self.cooldown_sec * 10)):
                if self._stop:
                    return
                time.sleep(0.1)

    def get_latest(self):
        with self.lock:
            return self.latest

    def close(self):
        self._stop = True


class DeepfakeGuardSystem:
    # Mirrors backend/app.py's Week 3 weights and five-signal shape exactly,
    # so main.py's local overlay and the dashboard's /scores value never
    # disagree. "lip_sync" is dropped as its own weighted signal (it isn't
    # part of the backend's contract) and instead folds into liveness
    # override logic, same as the 2D anti-spoof flag already did.
    WEIGHTS = {
        "identity": 0.25,
        "liveness": 0.25,
        "visual_deepfake": 0.20,
        "audio_spoof": 0.20,
        "injection": 0.10,
    }
    BACKEND_URL = "http://localhost:5000"

    @property
    def no_backend(self):
        return self.backend_sync.no_backend

    @no_backend.setter
    def no_backend(self, value):
        self.backend_sync.no_backend = value

    def post_to_backend(self, fusion_json_str):
        """Hands the fusion JSON off to BackendSync's background thread --
        never blocks. See BackendSync for why this changed from a direct
        synchronous requests.post()."""
        self.backend_sync.submit_fusion(fusion_json_str)

    def fetch_backend_trust(self):
        """Returns whatever BackendSync's background thread last fetched
        from GET /scores -- never blocks, never opens a socket itself."""
        return self.backend_sync.get_latest_trust()

    def __init__(self):
        print("=" * 60)
        print("  DEEPFAKEGUARD - WEEK 3 FUSION SYSTEM")
        print("  Vision (anti-spoof + liveness + deepfake) + Identity + Audio + Injection")
        print("=" * 60)

        self.vision_detector = FaceLandmarkDetector()
        self.liveness_engine = LivenessChallengeEngine()
        self.deepfake_detector = DeepfakeDetector()  # pass model_path="models/xxx.pth" once you have real weights
        self.lip_sync = LipSyncDetector(window_ms=200)
        self.audio_detector = AASISTDetector()

        # ---- background inference workers ----
        # score_frame() (SigLIP, 300-800ms) and compute_identity_score()
        # (ArcFace, 200-500ms) used to run inline in the camera loop and
        # were the cause of the freezing/donma -- every call to either one
        # stalled the whole 30fps window. Both now run on their own
        # background thread; the main loop only submits the newest frame
        # and reads back whatever finished last, never waiting on either.
        self.deepfake_worker = AsyncWorker(self.deepfake_detector.score_frame, name="deepfake")
        self.backend_sync = BackendSync(self.BACKEND_URL)

        self.session_id = f"session_{int(time.time())}"
        self.challenge_active = False
        self.challenge_cooldown = 0
        self.frame_count = 0
        self.use_audio = False
        self.use_dashboard_trust = True  # option 2 from the fix list: trust the backend's number, not a local recompute
        # Rolling buffer of recent per-frame spoof flags. A single noisy
        # frame (bad lighting moment, motion blur) shouldn't be able to
        # trigger a hard DENIED by itself -- we require a clear majority
        # over a short recent window before treating it as a real spoof.
        self.spoof_history = deque(maxlen=15)

        # ---- Identity matching (ArcFace via InsightFace) ----
        # Loads enrolled_face.json if present; identity_score falls back to
        # a low, clearly-labeled value if no enrollment exists yet, rather
        # than a hardcoded "85 = trusted" placeholder.
        self.identity_matcher = IdentityMatcher()
        self.identity_worker = AsyncWorker(self.identity_matcher.compute_identity_score, name="identity")

        # ---- HUMAN / AI verdict (classify_human_ai) ----
        # See classify_human_ai() below for the full priority order. These
        # thresholds are on the SigLIP fake-probability scale (0=real,
        # 1=fake) and are tunable from the CLI (--ai-threshold /
        # --human-threshold) once detector_eval.py suggests better values
        # for your data. self._last_verdict implements hysteresis: while
        # the score sits between the two thresholds, the banner keeps
        # showing whatever it last confidently decided instead of
        # flickering every frame.
        self.ai_threshold = 0.60
        self.human_threshold = 0.35
        self._last_verdict = None
        self._last_verdict_time = 0.0
        # If the score sits in the uncertain middle band for longer than
        # this, stop holding the old decisive verdict -- a "HUMAN" decided
        # once early in the session (e.g. off the presenter's own real
        # face) should not keep being shown indefinitely once a different
        # face/photo is in frame and the model hasn't re-confirmed it in a
        # while. ANALYZING is a more honest state than a stale answer.
        self.hysteresis_timeout_sec = 4.0
        self._printed_verdict = None

        # ---- Injection / virtual-camera detection ----
        # Runs on its own thread + cooldown so its blocking, camera-opening
        # timing check doesn't stall the main loop or fight for /dev/video0.
        self.security_monitor = None  # created in run() once camera_index is known

        # ---- Microphone (real audio, replacing mock random noise) ----
        self.mic = None  # created in run() based on user's audio choice

    # ---------------- fusion ----------------
    def compute_trust_score(self, scores):
        """Local fallback calculation -- used only for the on-screen OpenCV
        overlay when the backend is unreachable (--no-backend, or backend
        not running). Uses the SAME weights and SAME band thresholds/labels
        as backend/app.py's compute_trust() so the two never disagree when
        both are available. When the backend IS reachable, run() prefers
        fetch_backend_trust() over this method entirely."""
        weighted_sum, total_weight = 0.0, 0.0
        for key, weight in self.WEIGHTS.items():
            val = scores.get(key)
            if val is not None:
                weighted_sum += val * weight
                total_weight += weight

        if total_weight == 0:
            return {"trust_score": 50.0, "band": "suspicious",
                    "lowest_signal": "none", "lowest_score": 0}

        trust_score = weighted_sum / total_weight
        # Same 3-band system as backend/app.py's compute_trust(): >=80
        # trusted, >=50 suspicious, else fraud. No more local
        # TRUSTED/CAUTION/SUSPICIOUS/DENIED four-band system.
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

    # ---------------- HUMAN / AI verdict ----------------
    def classify_human_ai(self, injection_result, anti_spoof_2d_flag):
        """Single top-line HUMAN / AI DETECTED verdict for the banner.

        Priority order (strongest, least-ambiguous evidence first):
          1. Virtual camera / frame-injection detected (security_module) --
             someone is feeding a fake video source into the pipeline.
          2. Sustained 2D screen-replay spoof (face_detector + liveness),
             already temporally smoothed upstream so one noisy frame can't
             trigger this.
          3. The real deepfake classifier's own score, but ONLY once real
             (non-mock) inferences exist -- get_stage2_window() is empty
             until the model has actually run, so this never fires on
             torch.rand-style placeholder numbers.
        Between the two thresholds, or before enough samples exist, this
        keeps the previous verdict (hysteresis) rather than flapping the
        banner every frame on borderline scores.
        """
        if injection_result and injection_result.get("virtual_camera_detected"):
            verdict = ("AI DETECTED", "VIRTUAL CAMERA INJECTION", (0, 0, 255))
            self._last_verdict = verdict
            self._last_verdict_time = time.time()
            return verdict

        if anti_spoof_2d_flag:
            verdict = ("AI DETECTED", "SCREEN REPLAY (2D SPOOF)", (0, 0, 255))
            self._last_verdict = verdict
            self._last_verdict_time = time.time()
            return verdict

        backend = self.deepfake_detector.backend
        if backend not in ("siglip_hf", "efficientnet_local"):
            # No real model loaded -- refuse to ever say HUMAN or AI off a
            # mock/random score. This is deliberate: a wrong verdict on
            # stage is worse than an honest "not ready yet".
            return ("ANALYZING", "MODEL NOT LOADED (mock)", (0, 255, 255))

        stage2_scores = self.deepfake_detector.get_stage2_window()
        if len(stage2_scores) < 5:
            return ("ANALYZING", "GATHERING SAMPLES", (0, 255, 255))

        p = statistics.median(stage2_scores)
        if p >= self.ai_threshold:
            verdict = ("AI DETECTED", f"DEEPFAKE MODEL p={p:.2f}", (0, 0, 255))
            self._last_verdict = verdict
            self._last_verdict_time = time.time()
            return verdict
        if p <= self.human_threshold:
            verdict = ("HUMAN", f"p={p:.2f}", (0, 255, 0))
            self._last_verdict = verdict
            self._last_verdict_time = time.time()
            return verdict

        # Borderline -- hold the previous decisive verdict, but only for a
        # limited time. Past hysteresis_timeout_sec without a fresh
        # confirmation, an old verdict is more likely stale than still
        # true (someone else stepped into frame, lighting changed, etc.).
        if self._last_verdict is not None and (time.time() - self._last_verdict_time) < self.hysteresis_timeout_sec:
            return self._last_verdict
        return ("ANALYZING", f"UNCERTAIN p={p:.2f}", (0, 255, 255))

    def generate_json(self, liveness_payload, vision_payload, deepfake_result,
                       audio_result, sync_result, trust_data, smoothed_spoof_flag,
                       identity_result, injection_result):
        challenge = liveness_payload.get("challenge_result")
        challenge_passed = bool(challenge and challenge.get("challenge_passed"))

        spoof_analysis = vision_payload.get("spoof_analysis") or {}
        # Use the temporally-smoothed (majority-vote-over-recent-frames)
        # spoof flag for the actual verdict -- not the single-frame raw
        # heuristic, which is noisy. The raw per-frame value is still
        # available in spoof_analysis for debugging if needed.
        anti_spoof_2d_flag = smoothed_spoof_flag
        face_detected = liveness_payload.get("face_detected", False)
        multi_face = liveness_payload.get("multi_face", False)

        # Hard overrides -- these are NOT averaged into the trust score,
        # they force a denial outright. A weighted-average trust score can
        # still land in "GRANTED" territory even when one strong signal
        # (e.g. "this is a phone screen") screams spoof; that's not
        # acceptable for an access-control verdict. The backend now also
        # enforces this same override in /fusion (see backend/app.py), so
        # this local verdict and the dashboard's band will agree.
        hard_deny_reasons = []
        if not face_detected:
            hard_deny_reasons.append("no_face_detected")
        if multi_face:
            hard_deny_reasons.append("multiple_faces")
        if anti_spoof_2d_flag:
            hard_deny_reasons.append("2d_spoof_detected_sustained")
        if injection_result and injection_result.get("virtual_camera_detected"):
            hard_deny_reasons.append("virtual_camera_detected")

        if hard_deny_reasons:
            verdict = "ACCESS DENIED"
        else:
            verdict = "ACCESS GRANTED" if trust_data["band"] in ("trusted", "suspicious") else "ACCESS DENIED"

        data = {
            "module": "fusion",
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "face_detected": face_detected,
            "multi_face_alert": multi_face,
            "trust_score": trust_data["trust_score"],
            "trust_band": trust_data["band"],
            "lowest_signal": trust_data["lowest_signal"],
            "signals": {
                "liveness_challenge_passed": challenge_passed,
                "visual_deepfake_probability": (
                    deepfake_result["deepfake_probability"] if deepfake_result else None
                ),
                "audio_spoof_probability": (
                    audio_result["spoof_probability"] if audio_result else None
                ),
                "lip_sync_score": sync_result.get("lip_sync_score"),
                "anti_spoof_2d_raw": spoof_analysis.get("is_spoof", False),
                "anti_spoof_2d_sustained": anti_spoof_2d_flag,
                "identity_similarity": identity_result.get("similarity_score") if identity_result else None,
                "injection_risk_score": injection_result.get("injection_risk_score") if injection_result else None,
                "virtual_camera_detected": injection_result.get("virtual_camera_detected") if injection_result else None,
            },
            "hard_deny_reasons": hard_deny_reasons,
            "verdict": verdict,
            "challenge_result": challenge,
        }
        return json.dumps(data, indent=2)

    # ---------------- test helper ----------------
    def create_test_video(self, path="test_video.mp4", frames=150):
        print(f"[INFO] Creating synthetic test video: {path}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, 30.0, (640, 480))
        for i in range(frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(frame, (320, 240), 100, (200, 180, 160), -1)
            cv2.circle(frame, (290, 220), 15, (50, 50, 50), -1)
            cv2.circle(frame, (350, 220), 15, (50, 50, 50), -1)
            cv2.ellipse(frame, (320, 270), (40, 20), 0, 0, 180, (100, 50, 50), 3)
            noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(frame)
        out.release()
        print(f"[OK] {path} ready")
        return path

    def _candidate_backends(self):
        """Return a list of (backend_id, label) to try, in order, based on OS.
        Falls back to cv2.CAP_ANY (let OpenCV pick) last on every platform,
        so this degrades gracefully even on an OS we didn't special-case."""
        system = platform.system()
        backends = []
        if system == "Darwin":
            backends.append((cv2.CAP_AVFOUNDATION, "AVFoundation"))
        elif system == "Windows":
            backends.append((cv2.CAP_DSHOW, "DirectShow"))
            backends.append((cv2.CAP_MSMF, "Media Foundation"))
        elif system == "Linux":
            backends.append((cv2.CAP_V4L2, "V4L2"))
        backends.append((cv2.CAP_ANY, "auto"))
        return backends

    def _try_read_with_retry(self, cap, attempts=10, delay=0.1):
        """Some backends need a couple of frames to warm up before read()
        succeeds -- retry briefly instead of giving up on the first miss."""
        for _ in range(attempts):
            ret, frame = cap.read()
            if ret and frame is not None:
                return frame
            time.sleep(delay)
        return None

    def _open_camera(self):
        print("\nOpening camera...")
        print("(On Mac: System Settings -> Privacy & Security -> Camera -> enable Terminal)")
        print("(On Windows: Settings -> Privacy & security -> Camera -> allow desktop apps)")
        print("(On Linux: make sure your user is in the 'video' group and no other app has /dev/video0 open)")

        indices_to_try = [0, 1, 2]
        for backend_id, backend_label in self._candidate_backends():
            for idx in indices_to_try:
                cap = cv2.VideoCapture(idx, backend_id)
                if not cap.isOpened():
                    cap.release()
                    continue
                test_frame = self._try_read_with_retry(cap)
                if test_frame is None:
                    cap.release()
                    continue
                print(f"Camera OK | index {idx} | backend: {backend_label} | frame shape: {test_frame.shape}")
                self._camera_index = idx
                return cap

        print("\n" + "=" * 50)
        print("  CAMERA COULD NOT BE OPENED")
        print("=" * 50)
        print("Likely causes:")
        print("  - OS-level camera permission not granted to this terminal/app")
        print("  - Another app is using the camera (Zoom/Teams/Meet/Chrome/etc.)")
        print("  - No camera at index 0/1/2, or a driver issue")
        print("Try: closing other apps using the camera, checking OS camera")
        print("     privacy settings, and rerunning.")
        return None

    # ---------------- main loop ----------------
    def run(self):
        print("\n[1] Camera (live)")
        print("[2] Video file")
        print("[3] Auto-generated test video (no camera needed)")
        choice = input("\nSelect: ").strip()

        self.use_audio = input("Enable real microphone audio pipeline too? (y/N): ").strip().lower() == "y"

        self._camera_index = 0
        cap = None
        if choice == "1":
            cap = self._open_camera()
            if cap is None:
                return
        elif choice == "2":
            video_path = input("Path to video file: ").strip()
            if not os.path.exists(video_path):
                print(f"File not found: {video_path}")
                return
            cap = cv2.VideoCapture(video_path)
        elif choice == "3":
            test_path = "test_video.mp4"
            if not os.path.exists(test_path):
                self.create_test_video(test_path)
            cap = cv2.VideoCapture(test_path)
        else:
            print("Invalid choice")
            return

        if not cap.isOpened():
            print("Could not open video source!")
            return

        # SecurityMonitor no longer opens its own VideoCapture -- it gets
        # frame timestamps fed from this loop's own cap.read() calls (see
        # feed_frame() below), so there is exactly one capture on the
        # device. Only meaningful on a real live camera (choice == "1");
        # video-file/mock sources don't have a meaningful "camera timing"
        # to check.
        if choice == "1":
            self.security_monitor = SecurityMonitor(
                camera_index=self._camera_index, session_id=self.session_id
            )
        else:
            self.security_monitor = None

        if self.use_audio:
            self.mic = MicrophoneStream(sample_rate=16000, max_seconds=2.0)

        print("Source opened. Press ESC in the video window to quit.\n")
        prev_time = time.time()

        is_live_camera = (choice == "1")
        consecutive_failed_reads = 0
        MAX_CONSECUTIVE_FAILS = 30  # ~1s of retries before we give up on a live camera

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    if is_live_camera:
                        # A live camera can drop a frame transiently (backend
                        # hiccup, resolution/format negotiation, USB glitch).
                        # Don't treat a single failed read as "end of stream" --
                        # retry a bit before actually giving up.
                        consecutive_failed_reads += 1
                        if consecutive_failed_reads >= MAX_CONSECUTIVE_FAILS:
                            print("\n[INFO] Camera stopped returning frames")
                            break
                        time.sleep(0.03)
                        continue
                    else:
                        print("\n[INFO] End of stream")
                        break
                consecutive_failed_reads = 0

                if self.security_monitor:
                    self.security_monitor.feed_frame()

                frame = cv2.flip(frame, 1)
                self.frame_count += 1
                now = time.time()
                fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 30.0
                prev_time = now

                faces, vision_payload = self.vision_detector.process_frame(frame)
                _, liveness_payload = self.liveness_engine.process_frame(frame, faces)

                # ---- deepfake classifier (background thread) ----
                # .copy() because the main loop draws overlays onto `frame`
                # later in this same iteration -- without the copy, the
                # worker thread could be reading pixels mid-mutation.
                # Reuse the face box FaceLandmarkDetector already found
                # this frame (vision_payload["landmarks"]) instead of
                # running a second, independent Haar cascade inside the
                # deepfake detector -- one less redundant detection call,
                # and both modules score the same face region.
                lm = vision_payload.get("landmarks")
                face_box = tuple(int(v) for v in lm) if lm and len(lm) == 4 else None
                self.deepfake_worker.submit(frame.copy(), face_box)
                deepfake_result = self.deepfake_worker.get_latest()
                if deepfake_result is None:
                    # Worker hasn't finished its first frame yet.
                    deepfake_result = {
                        "deepfake_probability": 0.5, "confidence": "low",
                        "cascade_stage": 0, "model_backend": self.deepfake_detector.backend or "mock",
                        "frames_scored_last_30s": 0, "latency_ms_avg": 0.0,
                        "timestamp": time.time(), "skipped": True,
                    }

                # ---- real microphone audio (replaces mock random noise) ----
                if self.mic is not None:
                    recent_ms_samples = int(self.mic.sample_rate * 0.2)  # ~200ms for lip-sync
                    audio_snippet = self.mic.get_recent(recent_ms_samples)
                else:
                    # No mic configured/available -- feed silence rather than
                    # random noise. Silence still lets lip_sync fall back to
                    # its own "insufficient_data"/low-confidence handling
                    # instead of scoring meaningless random numbers as if
                    # they were real audio.
                    audio_snippet = np.zeros(1600, dtype=np.float32)

                mouth_crop = self.liveness_engine.get_mouth_crop(frame, faces)
                if mouth_crop is not None:
                    self.lip_sync.update_mouth_openness(mouth_crop)
                self.lip_sync.update_audio(audio_snippet)
                sync_result = self.lip_sync.compute_sync_score()

                audio_result = None
                if self.use_audio and self.mic is not None and self.frame_count % 30 == 0:
                    full_chunk = self.mic.get_recent(self.mic.sample_rate)  # last 1s, real mic audio
                    audio_result = self.audio_detector.ensemble_score(full_chunk)

                # ---- identity matching (ArcFace), submitted every ~15th
                # frame but read every frame (background thread) ----
                if self.frame_count % 15 == 0:
                    self.identity_worker.submit(frame.copy(), faces)
                identity_result = self.identity_worker.get_latest()

                # ---- injection / virtual camera (background thread, own cooldown) ----
                injection_result = self.security_monitor.get_latest() if self.security_monitor else None

                # ---- challenge issuing / display ----
                challenge = liveness_payload.get("challenge_result")
                if challenge:
                    passed = challenge.get("challenge_passed", False)
                    color = (0, 255, 0) if passed else (0, 0, 255)
                    text = f"CHALLENGE: {challenge['challenge_type']} - {'PASS' if passed else 'FAIL'}"
                    cv2.putText(frame, text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    # PASS və FAIL-dən sonra challenge bitsin
                    self.challenge_active = False
                    self.challenge_cooldown = self.frame_count + 60

                    # Engine reset
                    self.liveness_engine.reset()
                elif not self.challenge_active and self.frame_count > self.challenge_cooldown:
                    if liveness_payload.get("face_detected") and liveness_payload.get("state") == "IDLE":
                        new_challenge = self.liveness_engine.issue_challenge()
                        self.challenge_active = True
                        print(f"\n>>> NEW CHALLENGE: {new_challenge}")

                if self.challenge_active and liveness_payload.get("state") == "AWAITING_RESPONSE":
                    elapsed = time.time() - self.liveness_engine.challenge_start_time
                    remaining = max(0, int(10 - elapsed))
                    cv2.putText(frame, f"CHALLENGE: {liveness_payload.get('challenge_type', '')} | {remaining}s",
                                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                # ---- fusion ----
                challenge_passed = bool(challenge and challenge.get("challenge_passed"))
                spoof_analysis = vision_payload.get("spoof_analysis") or {}
                raw_spoof_flag = spoof_analysis.get("is_spoof", False) or liveness_payload.get("is_static", False)

                # Temporal smoothing: only treat this as a real spoof once a
                # clear majority of the last N frames agree. A single noisy
                # frame (bad lighting instant, motion blur) shouldn't be
                # able to deny a real user by itself.
                self.spoof_history.append(raw_spoof_flag)
                spoof_ratio = sum(self.spoof_history) / len(self.spoof_history)
                anti_spoof_2d_flag = spoof_ratio >= 0.6 and len(self.spoof_history) >= 5

                identity_trust = (identity_result["identity_score"]
                                   if identity_result else self.identity_matcher.last_score())
                injection_trust = (injection_result["injection_risk_score"]
                                    if injection_result else 50)  # neutral placeholder if not yet reported

                scores = {
                    "identity": identity_trust,
                    "liveness": (100 if challenge_passed else 50) if not anti_spoof_2d_flag else 20,
                    "visual_deepfake": 100 - self.deepfake_detector.get_visual_deepfake_score(),
                    "audio_spoof": 100 - (self.audio_detector.get_audio_spoof_score() if self.use_audio else 50),
                    "injection": injection_trust,
                }
                local_trust_data = self.compute_trust_score(scores)

                if self.frame_count % 30 == 0:
                    output = self.generate_json(liveness_payload, vision_payload, deepfake_result,
                                                 audio_result, sync_result, local_trust_data, anti_spoof_2d_flag,
                                                 identity_result, injection_result)
                    print(f"\n--- FUSION OUTPUT ---\n{output}")
                    self.post_to_backend(output)

                # Prefer the backend's own /scores number for the on-screen
                # overlay so the presenter's window and the projector never
                # show two different trust scores. Falls back to the local
                # calc if the backend is unreachable or disabled.
                trust_data = local_trust_data
                if self.use_dashboard_trust and self.frame_count % 15 == 0:
                    backend_scores = self.fetch_backend_trust()
                    if backend_scores is not None:
                        backend_trust = backend_scores.get("ema_trust_score")
                        if backend_trust is None:
                            backend_trust = backend_scores.get("trust_score")
                        if backend_trust is not None:
                            trust_data = {
                                "trust_score": backend_trust,
                                "band": backend_scores.get("trust_band", local_trust_data["band"]),
                                "lowest_signal": local_trust_data["lowest_signal"],
                                "lowest_score": local_trust_data["lowest_score"],
                            }
                            self._last_backend_trust_data = trust_data
                if trust_data is local_trust_data and hasattr(self, "_last_backend_trust_data") and self.use_dashboard_trust:
                    # Reuse last known backend value between the 15-frame polls
                    # instead of flickering back to the local number every frame.
                    trust_data = self._last_backend_trust_data

                # ---- rendering ----
                frame = self.vision_detector.draw_landmarks(frame, faces, vision_payload)

                trust = trust_data["trust_score"]
                band = trust_data["band"]
                band_color = {
                    "trusted": (0, 255, 0),
                    "suspicious": (0, 165, 255),
                    "fraud": (0, 0, 255),
                }.get(band, (0, 165, 255))

                bar_x, bar_y, bar_w, bar_h = 20, 180, 200, 20
                filled = int(bar_w * trust / 100)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), band_color, -1)
                cv2.putText(frame, f"TRUST: {trust:.0f} ({band.upper()})", (bar_x, bar_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, band_color, 2)
                cv2.putText(frame, f"Weak signal: {trust_data['lowest_signal']}", (bar_x, bar_y + bar_h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                # ---- verdict banner (same hard-deny rule as generate_json) ----
                face_detected = liveness_payload.get("face_detected", False)
                multi_face = liveness_payload.get("multi_face", False)
                virtual_cam = bool(injection_result and injection_result.get("virtual_camera_detected"))
                if not face_detected:
                    verdict_text, verdict_color = "ACCESS DENIED - NO FACE", (0, 0, 255)
                elif multi_face:
                    verdict_text, verdict_color = "ACCESS DENIED - MULTIPLE FACES", (0, 0, 255)
                elif anti_spoof_2d_flag:
                    verdict_text, verdict_color = "ACCESS DENIED - SPOOF DETECTED", (0, 0, 255)
                elif virtual_cam:
                    verdict_text, verdict_color = "ACCESS DENIED - VIRTUAL CAMERA", (0, 0, 255)
                elif band in ("trusted", "suspicious"):
                    verdict_text, verdict_color = "ACCESS GRANTED", (0, 255, 0)
                else:
                    verdict_text, verdict_color = "ACCESS DENIED", (0, 0, 255)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 4), verdict_color, -1)
                (tw, th), _ = cv2.getTextSize(verdict_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(frame, verdict_text, (frame.shape[1] - tw - 20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, verdict_color, 2)

                # ---- HUMAN / AI verdict (the main goal) ----
                # Small text line, not a big banner over the face -- same
                # style/size as the other status lines below, just bolder
                # and colored so it's easy to spot at a glance.
                ha_label, ha_reason, ha_color = self.classify_human_ai(injection_result, anti_spoof_2d_flag)
                cv2.putText(frame, f"VERDICT: {ha_label} ({ha_reason})", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, ha_color, 2)
                current_verdict_key = (ha_label, ha_reason)
                if current_verdict_key != self._printed_verdict:
                    print(f"[VERDICT] {ha_label} - {ha_reason}")
                    self._printed_verdict = current_verdict_key

                cv2.putText(frame, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                df_prob = deepfake_result.get("deepfake_probability", 0.0)
                cv2.putText(frame, f"Deepfake prob: {df_prob:.2f}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"State: {liveness_payload.get('state', 'IDLE')}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Blinks: {liveness_payload.get('blink_count', 0)}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                id_label = f"Identity: {identity_trust}/100" if identity_result or self.identity_matcher.enrolled else "Identity: NOT ENROLLED"
                cv2.putText(frame, id_label, (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if injection_result:
                    cv2.putText(frame, f"Injection risk: {injection_trust}/100 ({injection_result.get('device_name', '?')})",
                                (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                cv2.imshow("DeepfakeGuard - Week 3 Fusion", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("\n[INFO] Quit (ESC)")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.liveness_engine.close()
            self.vision_detector.close()
            self.deepfake_worker.close()
            self.identity_worker.close()
            self.backend_sync.close()
            if self.security_monitor:
                self.security_monitor.close()
            if self.mic:
                self.mic.close()
            print("\n[INFO] System shut down cleanly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepfakeGuard Fusion System")
    parser.add_argument("--no-backend", action="store_true",
                        help="Disable posting fusion data to the Flask backend")
    parser.add_argument("--ai-threshold", type=float, default=0.60,
                        help="SigLIP median score (0-1) at/above which the banner says AI DETECTED (default 0.60)")
    parser.add_argument("--human-threshold", type=float, default=0.35,
                        help="SigLIP median score (0-1) at/below which the banner says HUMAN (default 0.35)")
    parser.add_argument("--flip-deepfake-labels", action="store_true",
                        help="Swap which softmax index counts as 'fake' -- try this if HUMAN/AI results "
                             "look inverted (watch the [DEEPFAKE] p(fake_idx0)/p(real_idx1) debug line)")
    args = parser.parse_args()
    system = DeepfakeGuardSystem()
    system.no_backend = args.no_backend
    system.ai_threshold = args.ai_threshold
    system.human_threshold = args.human_threshold
    system.deepfake_detector.flip_labels = args.flip_deepfake_labels
    system.run()
