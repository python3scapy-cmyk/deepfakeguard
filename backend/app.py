from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
from datetime import datetime, timezone
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import base64
import json
import random
import threading
import tempfile
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import send_from_directory
from vision.deepfake_detector import DeepfakeDetector
from engine import get_engine, drop_session
from security.device_check import classify_camera_label
app = Flask(__name__)
# max_http_buffer_size: engine.io defaults to 1 MB. A 6s voice clip sent as
# 48kHz float32 base64 was ~1.5 MB, so the websocket was closed with
# "Invalid frame header" and the clip never arrived - the voice challenge
# then timed out even though the browser had recorded perfectly good audio.
# The client now downsamples to 16kHz int16 (~256 KB), and this ceiling
# keeps a margin.
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10_000_000)
CORS(app)  # allows cross-origin HTTP requests from the dashboard

# ─────────────────────────────────────────────
# In-memory store — one entry per module,
# always the latest payload received.
# ─────────────────────────────────────────────
SIGNAL_TIMEOUT_SEC = 3.0   # if no report in 3s, mark as missing
MAX_CHALLENGE_HISTORY = 20
HARD_DENY_TIMEOUT_SEC = 3.0
MAX_LOG_ENTRIES = 500
EMA_ALPHA = 0.3


# ─────────────────────────────────────────────
# Rooms & per-client state
#
# The dashboard used to read ONE global set of module scores, so two people
# verifying at the same time overwrote each other. State is now sharded:
#
#   ROOMS[code].clients[session_id] -> ClientState
#
# An admin creates a room (6-char code, PIN-gated) and shares it; each
# client that joins gets its own ClientState, so panels, logs, EMA and
# hard-deny timers are per-person. Requests without a code (main.py's
# /fusion POST, the legacy /score endpoints) fall back to the LOCAL room so
# the existing local-camera pipeline keeps working untouched.
# ─────────────────────────────────────────────
# Admin PIN. On first run there is NO pin: the first person to open /admin
# sets one through the UI, and it is persisted so it survives restarts.
# DFG_ADMIN_PIN, if set, overrides everything (useful on demo day).
ADMIN_PIN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "admin_pin.json")
_pin_lock = threading.Lock()


def _load_pin():
    env = os.environ.get("DFG_ADMIN_PIN")
    if env:
        return str(env)
    try:
        with open(ADMIN_PIN_FILE, "r") as f:
            return str(json.load(f).get("pin") or "") or None
    except Exception:
        return None


def _save_pin(pin):
    with open(ADMIN_PIN_FILE, "w") as f:
        json.dump({"pin": str(pin)}, f)


def admin_pin():
    return _load_pin()


def _coded_room_write_blocked(code):
    """Guard for state-WRITING HTTP endpoints (/score, /session-reset,
    /analyze-upload). Returns an error string if the request targets a
    CODED room without the admin PIN, else None.

    Rationale: these endpoints were fully unauthenticated, so any
    participant who knew their own room code + session_id could POST
    whitewashing module scores or reset a session mid-verification. The
    ?demo=1 gate on the dashboard is client-side only and protects
    nothing. The LOCAL room stays open - main.py's loopback pipeline
    must keep working with zero changes - and coded rooms now require
    the admin PIN via the X-Admin-Pin header (or ?pin= for curl).
    index.html sends the header on every affected call after login."""
    if not code or str(code).upper() == DEFAULT_CODE:
        return None                      # LOCAL room: unchanged, open
    current = admin_pin()
    supplied = request.headers.get("X-Admin-Pin") or request.args.get("pin") or ""
    if current is not None and str(supplied) == str(current):
        return None
    return "writes to coded rooms require the admin PIN (X-Admin-Pin header)"


DEFAULT_CODE = "LOCAL"
LOCAL_SESSION = "local"
ROOM_IDLE_SEC = 30 * 60
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no look-alikes (0/O, 1/I)


class ClientState:
    """Everything the dashboard renders for ONE verifying person."""

    def __init__(self, session_id):
        self.session_id = session_id
        self.latest_scores = {"vision": None, "audio": None, "security": None,
                              "identity": None, "visual_deepfake": None}
        self.last_seen = {"vision": None, "audio": None, "security": None,
                          "identity": None, "visual_deepfake": None}
        self.vision_challenge_history = []
        self._latest_hard_deny_reasons = []
        self._hard_deny_last_seen = None
        self.session_log = []
        self._last_trust_band = None
        self._ema_trust_score = None
        self.created_at = time.time()
        self.last_active = time.time()
        # Rolling summary for the admin's client list
        self.summary = {"trust_score": None, "band": None, "verdict": None,
                        "session_final": False}
        # Camera identity of this client. NAME comes from the browser
        # (only place that can read the MediaStreamTrack label); CLASS is
        # re-derived SERVER-SIDE via security/device_check.py, so the
        # client's own opinion of its class is never trusted for the
        # verdict. 'virtual' -> hard deny, 'suspect' -> penalty (see
        # on_analysis_frame).
        self.camera_name = None
        self.camera_class = None

    def touch(self):
        self.last_active = time.time()


class Room:
    def __init__(self, code):
        self.code = code
        self.created_at = time.time()
        self.last_active = time.time()
        self.clients = {}          # session_id -> ClientState

    def client(self, session_id):
        st = self.clients.get(session_id)
        if st is None:
            st = ClientState(session_id)
            self.clients[session_id] = st
            print(f"[ROOM {self.code}] client joined: {session_id} "
                  f"({len(self.clients)} in room)")
        self.last_active = time.time()
        st.touch()
        return st

    def most_recent_client(self):
        if not self.clients:
            return None
        return max(self.clients.values(), key=lambda c: c.last_active)


ROOMS = {DEFAULT_CODE: Room(DEFAULT_CODE)}
_rooms_lock = threading.Lock()


def new_code():
    with _rooms_lock:
        while True:
            code = "".join(random.choice(CODE_ALPHABET) for _ in range(6))
            if code not in ROOMS:
                ROOMS[code] = Room(code)
                print(f"[ROOM {code}] created")
                return code


def get_room(code, create=False):
    code = (code or DEFAULT_CODE).upper()
    with _rooms_lock:
        room = ROOMS.get(code)
        if room is None and (create or code == DEFAULT_CODE):
            room = Room(code)
            ROOMS[code] = room
        return room


def prune_rooms():
    now = time.time()
    with _rooms_lock:
        for code in [c for c, r in ROOMS.items()
                     if c != DEFAULT_CODE and now - r.last_active > ROOM_IDLE_SEC]:
            del ROOMS[code]
            print(f"[ROOM {code}] pruned (idle)")


def resolve_state(code=None, session_id=None, create=False):
    """Room + client resolution used by every request/event entry point."""
    room = get_room(code, create=create)
    if room is None:
        return None, None
    if session_id:
        return room, room.client(session_id)
    st = room.most_recent_client()
    if st is None:
        st = room.client(LOCAL_SESSION if room.code == DEFAULT_CODE else "pending")
    return room, st


app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024  # 150MB cap on uploads
# Faza 3: the standalone upload detector is gone - /analyze-upload now
# runs the FULL pipeline via engine.analyze_file(), which reuses the
# engine's shared SigLIP instance instead of loading a second ~370MB copy.
ALLOWED_VIDEO_EXT = {"mp4", "mov", "avi", "webm", "mkv"}
ALLOWED_AUDIO_EXT = {"wav", "mp3", "m4a", "flac", "ogg", "aac"}
ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "bmp"}
MAX_UPLOAD_FRAMES = 20  # mirrors the "Sequence Length" slider in your reference demo

def log_event(state, event_type, detail, band=None):
    state.session_log.append({
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "event_type":  event_type,   # e.g. "liveness_challenge", "identity_check", "band_change"
        "detail":      detail,
        "band":        band
    })
    if len(state.session_log) > MAX_LOG_ENTRIES:
        state.session_log.pop(0)

REQUIRED_FIELDS = {
    "vision":   ["module", "session_id", "timestamp", "payload"],
    "audio":    ["module", "session_id", "timestamp", "voice_detected",
                 "audio_deepfake_score", "verdict"],
    "security": ["module", "session_id", "timestamp",
                 "virtual_camera_detected", "injection_risk_score",
                 "frame_timing_anomaly_score", "verdict"],
    "identity": ["module", "session_id", "timestamp", "face_detected",
                 "multiple_faces_detected", "identity_score",
                 "similarity_score", "threshold_used", "confidence", "verdict"],
    "visual_deepfake": ["module", "session_id", "timestamp",
                         "deepfake_probability", "confidence", "skipped"]
}

VALID_CONFIDENCE_LEVELS = {"high", "low", "none"}
# visual_deepfake's confidence has no "none" state per M1's contract —
# it's always high or low based on rolling-window agreement.
VALID_VISUAL_CONFIDENCE_LEVELS = {"high", "low"}

# Fields required inside vision's nested "payload" object, per M1's
# confirmed contract (challenge_type, prompt, timeout_sec,
# challenge_passed, fail_reason, metrics.response_time_ms).
VISION_PAYLOAD_FIELDS = [
    "challenge_type", "prompt", "timeout_sec",
    "challenge_passed", "fail_reason", "metrics"
]

VALID_CHALLENGE_TYPES = {"blink_twice", "turn_left", "turn_right"}
VALID_FAIL_REASONS = {"timeout", "wrong_direction", "spoof_detected", "multiple_faces_detected"}

SCORE_FIELDS = {
    "vision":   [],  # vision is validated separately via VISION_PAYLOAD_FIELDS
    "audio":    ["audio_deepfake_score"],
    "security": ["injection_risk_score", "frame_timing_anomaly_score"],
    "identity": ["identity_score"],
    "visual_deepfake": []  # validated separately via validate_visual_deepfake_payload
}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def validate_payload(payload, module):
    """
    Returns (True, None) if payload is valid,
    or (False, error_message) if not.
    """
    for field in REQUIRED_FIELDS[module]:
        if field not in payload:
            return False, "missing required field: '{}'".format(field)

    if module == "vision":
        return validate_vision_payload(payload)

    if module == "visual_deepfake":
        return validate_visual_deepfake_payload(payload)

    for field in SCORE_FIELDS[module]:
        val = payload.get(field)
        if not isinstance(val, (int, float)):
            return False, "field '{}' must be a number".format(field)
        if not (0 <= val <= 100):
            return False, "field '{}' must be 0–100, got {}".format(field, val)

    if module == "identity":
        if payload.get("confidence") not in VALID_CONFIDENCE_LEVELS:
            return False, "confidence must be one of: {}".format(", ".join(VALID_CONFIDENCE_LEVELS))
        if not isinstance(payload.get("face_detected"), bool):
            return False, "face_detected must be a boolean"
        if not isinstance(payload.get("multiple_faces_detected"), bool):
            return False, "multiple_faces_detected must be a boolean"

    if payload.get("verdict") not in ("REAL", "FAKE"):
        return False, "verdict must be 'REAL' or 'FAKE'"

    return True, None


def validate_vision_payload(payload):
    """
    Validates M1's nested liveness-challenge payload shape:
    { module, session_id, timestamp, payload: { challenge_type, prompt,
      timeout_sec, challenge_passed, fail_reason, metrics: { response_time_ms, ... } } }
    """
    inner = payload.get("payload")
    if not isinstance(inner, dict):
        return False, "vision 'payload' must be an object"

    for field in VISION_PAYLOAD_FIELDS:
        if field not in inner:
            return False, "missing required vision.payload field: '{}'".format(field)

    if inner["challenge_type"] not in VALID_CHALLENGE_TYPES:
        return False, "invalid challenge_type: '{}'".format(inner["challenge_type"])

    if not isinstance(inner["challenge_passed"], bool):
        return False, "challenge_passed must be a boolean"

    fail_reason = inner.get("fail_reason")
    if inner["challenge_passed"]:
        if fail_reason is not None:
            return False, "fail_reason must be null when challenge_passed is true"
    else:
        if fail_reason not in VALID_FAIL_REASONS:
            return False, "invalid fail_reason: '{}'".format(fail_reason)

    metrics = inner.get("metrics")
    if not isinstance(metrics, dict):
        return False, "vision.payload.metrics must be an object"
    if not isinstance(metrics.get("response_time_ms"), (int, float)):
        return False, "metrics.response_time_ms must be a number"

    return True, None


def validate_visual_deepfake_payload(payload):
    """
    Validates M1's confirmed visual_deepfake contract:
    { module: "visual_deepfake", session_id, timestamp,
      deepfake_probability (0.0-1.0, higher = more fake), confidence
      ("high"/"low"), cascade_stage, frames_scored_last_30s,
      latency_ms_avg, skipped (bool) }

    Note: deepfake_probability is 0.0-1.0, NOT 0-100 like every other
    module's score fields — do not run it through the generic SCORE_FIELDS
    0-100 check.
    """
    prob = payload.get("deepfake_probability")
    if not isinstance(prob, (int, float)):
        return False, "deepfake_probability must be a number"
    if not (0.0 <= prob <= 1.0):
        return False, "deepfake_probability must be 0.0-1.0, got {}".format(prob)

    if payload.get("confidence") not in VALID_VISUAL_CONFIDENCE_LEVELS:
        return False, "visual_deepfake confidence must be one of: {}".format(
            ", ".join(VALID_VISUAL_CONFIDENCE_LEVELS))

    if not isinstance(payload.get("skipped"), bool):
        return False, "skipped must be a boolean"

    return True, None


def score_to_tier(score):
    """Maps a 0–100 score to a trust tier string."""
    if score >= 80:
        return "good"
    elif score >= 50:
        return "warn"
    return "bad"


def derive_liveness_score(vision_inner):
    """
    Converts M1's pass/fail challenge event into a 0-100 score so it can
    still feed compute_trust()'s weighted-sum fusion. This is a simple,
    tunable mapping, not something M1 needs to emit itself.
    """
    if vision_inner is None:
        return None
    if vision_inner.get("challenge_passed"):
        return 95
    # Different failure reasons could eventually carry different
    # severity; for now all failures score low and let fail_reason
    # carry the nuance in the dashboard's explainability text.
    return 15


def normalize_to_trust_contribution(field, raw_score, payload=None):
    """
    Convert a raw 0-100 score (or None) into a 0.0-1.0 trust contribution.
    Missing signals use a neutral 0.5 placeholder.
    """
    if raw_score is None:
        return 0.5  # neutral placeholder for missing signals

    # All scores are 0-100, higher = more trustworthy
    trust = raw_score / 100.0

    # Special handling for injection: if virtual camera detected, cap at 0.2
    if field == "injection_risk_score" and payload and payload.get("virtual_camera_detected"):
        trust = min(trust, 0.2)

    return round(trust, 2)


def get_active_hard_deny_reasons(state):
    """Returns the current hard-deny reasons from main.py's fusion verdict,
    but only if they're still fresh (within HARD_DENY_TIMEOUT_SEC). A stale
    hard-deny (main.py stopped posting, or the condition cleared several
    seconds ago) should not permanently pin the session at fraud."""
    if state._hard_deny_last_seen is None:
        return []
    if (time.time() - state._hard_deny_last_seen) > HARD_DENY_TIMEOUT_SEC:
        return []
    return state._latest_hard_deny_reasons


# Conflict penalties (deducted from trust score) -- parity with
# security/fusion_engine.py's CONFLICT_PENALTIES.
CONFLICT_PENALTIES = {
    "identity_vs_visual_deepfake": 0.10,
    "liveness_vs_injection":       0.30,
}

# Security hardening: minimum identity trust (0.0-1.0) to reach "trusted".
IDENTITY_TRUST_THRESHOLD = 0.60


def compute_trust(state, scores=None, sub_scores=None, conflicts=None):
    """Weighted fusion + hard-deny + conflict penalty + EMA.

    `state` carries the per-client history (EMA, hard-deny timers). It is
    OPTIONAL so the pure-function parity test in tests/test_fusion_parity.py
    can still call compute_trust(scores, sub_scores, conflicts) with no
    session context: when the first argument is a dict, it is treated as
    `scores` and a throwaway ClientState supplies the history slots.
    """
    if isinstance(state, dict):            # called without a state -> shift args
        state, scores, sub_scores, conflicts = (
            ClientState("_stateless"), state, scores, sub_scores)

    # Order of operations (must match security/fusion_engine.py exactly):
    #   weighted sum -> conflict penalty -> identity cap -> hard-deny
    #   override -> band
    weights = {
        "identity_score":        0.25,
        "liveness_score":        0.25,
        "visual_deepfake_score": 0.20,
        "audio_deepfake_score":  0.20,
        "injection_risk_score":  0.10
    }

    # NOTE: use sum() over a generator, NOT a manual += loop.
    # Python 3.12's sum() uses compensated summation for floats;
    # a manual loop doesn't get that and can be off-by-one at
    # boundary values (e.g. 0.5 exactly).
    weighted_sum = sum(sub_scores.get(field, 0.5) * weight for field, weight in weights.items())

    # NOTE: int() truncation here, NOT round() -- must match
    # fusion_engine.py's compute_trust_score() exactly, or the two
    # engines drift apart after the conflict-penalty multiply below.
    trust = int(weighted_sum * 100)
    trust = max(0, min(100, trust))

    # ---- Conflict penalty ----
    conflict_types = (conflicts or {}).get("conflict_types", [])
    conflict_penalty_pct = 0.0
    if conflict_types:
        conflict_penalty_pct = max(CONFLICT_PENALTIES.get(c, 0.0) for c in conflict_types)
        trust = int(trust * (1 - conflict_penalty_pct))
        trust = max(0, min(100, trust))

    # ---- Identity threshold hardening (cap at 79 / suspicious) ----
    identity_capped = False
    identity_trust = sub_scores.get("identity_score", 0.5)
    if identity_trust < IDENTITY_TRUST_THRESHOLD and trust >= 80:
        trust = 79
        identity_capped = True

    # ---- Hard-deny override, then band ----
    hard_deny_reasons = get_active_hard_deny_reasons(state)
    if hard_deny_reasons:
        band = "fraud"
    elif trust >= 80:
        band = "trusted"
    elif trust >= 50:
        band = "suspicious"
    else:
        band = "fraud"

    return trust, band, hard_deny_reasons, identity_capped, conflict_penalty_pct


def apply_ema(state, new_trust):
    """
    Apply Exponential Moving Average smoothing to the trust score.
    alpha = 0.3 means responsive but filtered.
    """
    if state._ema_trust_score is None:
        state._ema_trust_score = float(new_trust)
    else:
        state._ema_trust_score = EMA_ALPHA * new_trust + (1 - EMA_ALPHA) * state._ema_trust_score
    return round(state._ema_trust_score)


def detect_conflicts(flat_scores):
    """
    Detects alarming signal combinations that indicate a sophisticated
    attacker who fooled one layer but not another.
    """
    conflicts = []
    identity = flat_scores.get("identity_score")
    visual = flat_scores.get("visual_deepfake_score")
    liveness = flat_scores.get("liveness_score")
    injection = flat_scores.get("injection_risk_score")

    if identity is not None and visual is not None:
        if identity > 70 and visual < 30:
            conflicts.append("identity_vs_visual_deepfake")

    if liveness is not None and injection is not None:
        if liveness > 80 and injection < 30:
            conflicts.append("liveness_vs_injection")

    return {"conflict_detected": len(conflicts) > 0, "conflict_types": conflicts}


HARD_DENY_DISPLAY = {
    "no_face_detected":            "no face detected in frame",
    "multiple_faces":              "multiple faces detected in frame",
    "2d_spoof_detected_sustained": "sustained 2D spoof pattern detected (printed photo or screen replay)",
    "virtual_camera_detected":     "virtual camera / injected video source detected",
}


def build_reason(scores, verdicts, vision_fail_reason=None, identity_confidence=None,
                 conflicts=None, hard_deny_reasons=None, identity_capped=False,
                 conflict_penalty_pct=0.0):
    """
    Builds a human-readable reason string based on
    which signals are flagged, for the dashboard
    explainability area.

    Hard-deny reasons are surfaced FIRST (fix #2) since they're the actual
    reason the fused verdict is "fraud" when a hard-deny is active --
    burying them below softer score-based flags would misrepresent why
    the session was denied.
    """
    flags = []

    if hard_deny_reasons:
        for reason in hard_deny_reasons:
            flags.append("hard block — " + HARD_DENY_DISPLAY.get(reason, reason.replace("_", " ")))

    if conflicts and conflicts.get("conflict_detected"):
        for ct in conflicts.get("conflict_types", []):
            if ct == "identity_vs_visual_deepfake":
                flags.append("identity matched but visual deepfake risk is elevated")
            elif ct == "liveness_vs_injection":
                flags.append("liveness passed but injection risk is high")

    if scores.get("injection_risk_score") is not None:
        if scores["injection_risk_score"] < 50:
            flags.append("active injection attempt detected")
        elif scores["injection_risk_score"] < 80:
            flags.append("elevated injection-risk signal")

    if scores.get("visual_deepfake_score") is not None:
        if scores["visual_deepfake_score"] < 50:
            flags.append("deepfake artifacts confirmed in video stream")
        elif scores["visual_deepfake_score"] < 80:
            flags.append("minor visual artifacts detected")

    if scores.get("audio_deepfake_score") is not None:
        if scores["audio_deepfake_score"] < 50:
            flags.append("synthetic voice confirmed")
        elif scores["audio_deepfake_score"] < 80:
            flags.append("anomalous audio patterns in frequency analysis")

    if scores.get("liveness_score") is not None:
        if scores["liveness_score"] < 50:
            reason_text = "liveness challenge failed"
            if vision_fail_reason:
                reason_text += " ({})".format(vision_fail_reason.replace("_", " "))
            flags.append(reason_text)
        elif scores["liveness_score"] < 80:
            flags.append("liveness confidence below threshold")

    if scores.get("identity_score") is not None:
        if scores["identity_score"] < 50:
            flags.append("identity mismatch detected")
        elif identity_confidence == "low":
            flags.append("identity match in uncertain confidence zone")

    if "FAKE" in verdicts:
        flags.insert(0, "one or more modules returned FAKE verdict")

    if not flags:
        base_reason = "All signals within normal thresholds. Biometric match confirmed."
    else:
        base_reason = "Score reduced — " + "; ".join(flags) + "."

    if identity_capped:
        base_reason += " [Identity match below threshold — capped at Suspicious]"

    if conflict_penalty_pct:
        base_reason += " [Conflict penalty: -{}% applied]".format(int(conflict_penalty_pct * 100))

    return base_reason


def get_missing_signals(state):
    """
    Returns a list of module names that have not reported
    within SIGNAL_TIMEOUT_SEC seconds.
    """
    now = time.time()
    missing = []
    for module, last in state.last_seen.items():
        if last is None or (now - last) > SIGNAL_TIMEOUT_SEC:
            missing.append(module)
    return missing

def _allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    ok = (ext in ALLOWED_VIDEO_EXT or ext in ALLOWED_IMAGE_EXT
          or ext in ALLOWED_AUDIO_EXT)
    return ok, ext


@app.route('/analyze-upload', methods=['POST'])
def analyze_upload():
    """Faza 3: uploads now run the FULL pipeline (visual deepfake + 2D
    spoof heuristics + identity continuity + audio spoof via ffmpeg),
    fused with the same weights/bands as the live path. Legacy response
    fields are preserved; a 'full_pipeline' block is added on top."""
    upload_code = request.args.get("code")
    err = _coded_room_write_blocked(upload_code)
    if err:
        return jsonify({"error": err}), 401
    _room, state = resolve_state(upload_code, request.args.get("session"),
                                 create=(not upload_code
                                         or str(upload_code).upper() == DEFAULT_CODE))
    if state is None:
        return jsonify({"error": "unknown room code"}), 404
    if "media" not in request.files:
        return jsonify({"error": "no file field 'media' in request"}), 400
    f = request.files["media"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    ok, ext = _allowed_file(f.filename)
    if not ok:
        return jsonify({"error": "unsupported file type: .{}".format(ext)}), 400

    filename = secure_filename(f.filename)
    tmp_dir = tempfile.mkdtemp(prefix="dfg_upload_")
    tmp_path = os.path.join(tmp_dir, filename)
    f.save(tmp_path)

    is_video = ext in ALLOWED_VIDEO_EXT
    is_audio = ext in ALLOWED_AUDIO_EXT
    try:
        from engine import analyze_file, analyze_audio_file
        if is_audio:
            result = analyze_audio_file(tmp_path)
        else:
            result = analyze_file(tmp_path, is_video=is_video,
                                  max_frames=MAX_UPLOAD_FRAMES)
    except Exception as e:
        # Surface engine crashes instead of a bare 500 with no clue why.
        import traceback
        print("[ANALYZE_UPLOAD ERROR] " + str(e))
        print(traceback.format_exc())
        return jsonify({"error": "analysis engine crashed: " + str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass

    if "error" in result:
        return jsonify(result), 400

    fp = result.get("full_pipeline") or {}
    if fp.get("hard_deny_reasons"):
        log_event(state, "upload_analysis",
                   "Upload flagged - {}".format(
                       ", ".join(fp["hard_deny_reasons"]).replace("_", " ")))
    return jsonify(result)

# ─────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules_reporting": [k for k, v in state.latest_scores.items() if v is not None]
    })


@app.route('/score', methods=['POST'])
def receive_score():
    payload = request.get_json(silent=True)

    if not payload:
        return jsonify({"error": "empty or invalid JSON body"}), 400

    # Room/client scoping. Legacy posters (main.py) send no code -> LOCAL
    # room, so nothing about the local pipeline changes. Coded rooms now
    # require the admin PIN and must already EXIST - /score no longer
    # silently creates rooms, which used to bypass the PIN-gated
    # /api/session/create entirely.
    code = payload.get("room_code") or request.args.get("code")
    err = _coded_room_write_blocked(code)
    if err:
        return jsonify({"error": err}), 401
    _room, state = resolve_state(code, payload.get("session_id"),
                                 create=(not code or str(code).upper() == DEFAULT_CODE))
    if state is None:
        return jsonify({"error": "unknown room code", "code": code}), 404

    module = payload.get("module")
    if module not in state.latest_scores:
        return jsonify({
            "error": "unknown module '{}', expected one of: vision, audio, security, identity, visual_deepfake".format(module)
        }), 400

    valid, err = validate_payload(payload, module)
    if not valid:
        return jsonify({"error": err, "module": module}), 400

    state.latest_scores[module] = payload
    state.last_seen[module] = time.time()

    if module == "vision":
        state.vision_challenge_history.append(payload)
        if len(state.vision_challenge_history) > MAX_CHALLENGE_HISTORY:
            state.vision_challenge_history.pop(0)
        inner = payload["payload"]
        print("[{}] received vision challenge — type: {}, passed: {}, reason: {}".format(
            datetime.now(timezone.utc).strftime("%H:%M:%S"),
            inner.get("challenge_type"),
            inner.get("challenge_passed"),
            inner.get("fail_reason")
        ))
        type_name = inner.get("challenge_type", "").replace("_", " ")
        if inner.get("challenge_passed"):
            log_event(state, "liveness_challenge",
                       "Liveness challenge \"{}\" passed ({} ms)".format(
                           type_name, inner.get("metrics", {}).get("response_time_ms")))
        else:
            log_event(state, "liveness_challenge",
                       "Liveness challenge \"{}\" FAILED — {}".format(
                           type_name, (inner.get("fail_reason") or "").replace("_", " ")))
    elif module == "identity":
        print("[{}] received identity payload — score: {}, confidence: {}, verdict: {}".format(
            datetime.now(timezone.utc).strftime("%H:%M:%S"),
            payload.get("identity_score"),
            payload.get("confidence"),
            payload.get("verdict")
        ))
        log_event(state, "identity_check",
                   "Identity check — {}% similarity, confidence: {}, verdict: {}".format(
                       round(payload.get("similarity_score", 0) * 100, 1),
                       payload.get("confidence"),
                       payload.get("verdict")))
    elif module == "visual_deepfake":
        # Skipped frames (frame-subsampling misses) are not worth
        # spamming into the session log — only log real scored frames.
        if not payload.get("skipped"):
            print("[{}] received visual_deepfake payload — probability: {}, confidence: {}".format(
                datetime.now(timezone.utc).strftime("%H:%M:%S"),
                payload.get("deepfake_probability"),
                payload.get("confidence")
            ))
            prob = payload.get("deepfake_probability", 0)
            if prob > 0.7 and payload.get("confidence") == "high":
                log_event(state, "visual_deepfake_alert",
                           "Visual deepfake probability elevated ({}%, high confidence)".format(
                               round(prob * 100, 1)))
    else:
        print("[{}] received {} payload — verdict: {}".format(
            datetime.now(timezone.utc).strftime("%H:%M:%S"),
            module,
            payload.get("verdict")
        ))
        if module == "security" and payload.get("virtual_camera_detected"):
            log_event(state, "injection_alert", "Virtual camera / injection signature detected")
        elif module == "audio" and payload.get("verdict") == "FAKE":
            log_event(state, "audio_alert", "Audio deepfake signal flagged")

    return jsonify({"received": True, "module": module})


def ingest_fusion(payload, state):
    """Maps a complete fusion payload dict into the state.latest_scores store.

    Called from TWO places (Faza 1):
      - the /fusion HTTP route below (main.py's local-camera POST)
      - the 'analysis_frame' Socket.IO handler (in-process AnalysisEngine
        results for browser participants)
    so the dashboard sees identical data regardless of the source."""
    now_ts = datetime.now(timezone.utc).isoformat()
    session_id = payload.get("session_id", "unknown")
    signals = payload.get("signals", {})

    # -- Map fusion signals into state.latest_scores --

    # Vision / liveness: map from challenge_result if available
    challenge = payload.get("challenge_result")
    if challenge and isinstance(challenge, dict):
        challenge_type = challenge.get("challenge_type", "blink_twice")
        challenge_passed = challenge.get("challenge_passed", False)
        fail_reason = challenge.get("fail_reason")
        response_time_ms = challenge.get("response_time_ms", 0)
        vision_payload = {
            "module": "vision",
            "session_id": session_id,
            "timestamp": now_ts,
            "payload": {
                "challenge_type": challenge_type,
                "prompt": "Please " + challenge_type.replace("_", " "),
                "timeout_sec": 10.0,
                "challenge_passed": challenge_passed,
                "fail_reason": fail_reason,
                "metrics": {
                    "response_time_ms": response_time_ms
                }
            }
        }
        state.latest_scores["vision"] = vision_payload
        state.last_seen["vision"] = time.time()
        state.vision_challenge_history.append(vision_payload)
        if len(state.vision_challenge_history) > MAX_CHALLENGE_HISTORY:
            state.vision_challenge_history.pop(0)
        type_name = challenge_type.replace("_", " ")
        if challenge_passed:
            log_event(state, "liveness_challenge",
                       "Liveness challenge \"{}\" passed ({} ms)".format(
                           type_name, response_time_ms))
        else:
            log_event(state, "liveness_challenge",
                       "Liveness challenge \"{}\" FAILED - {}".format(
                           type_name, (fail_reason or "").replace("_", " ")))

    # Identity: prefer main.py's real ArcFace signal (signals.identity_similarity),
    # since main.py now computes this via security.identity.IdentityMatcher
    # instead of a hardcoded constant. Falls back to the old trust_score-derived
    # approximation only if main.py hasn't sent a real identity signal yet.
    trust_score = payload.get("trust_score", 50)
    face_detected = payload.get("face_detected", False)
    multi_face = payload.get("multi_face_alert", False)

    identity_similarity = signals.get("identity_similarity")
    if identity_similarity is not None:
        similarity_raw = float(identity_similarity)
        identity_raw = similarity_raw * 100
    else:
        identity_raw = trust_score * 0.85 if face_detected else 20
        similarity_raw = min(identity_raw / 100.0, 1.0)

    if identity_raw >= 70:
        id_confidence = "high"
        id_verdict = "REAL"
    elif identity_raw >= 50:
        id_confidence = "low"
        id_verdict = "FAKE"
    else:
        id_confidence = "none"
        id_verdict = "FAKE"

    identity_payload = {
        "module": "identity",
        "session_id": session_id,
        "timestamp": now_ts,
        "face_detected": face_detected,
        "multiple_faces_detected": multi_face,
        "identity_score": round(identity_raw),
        "similarity_score": round(similarity_raw, 3),
        "threshold_used": 0.6,
        "confidence": id_confidence,
        "verdict": id_verdict
    }
    state.latest_scores["identity"] = identity_payload
    state.last_seen["identity"] = time.time()

    # Visual deepfake: use deepfake_probability from signals
    deepfake_prob = signals.get("visual_deepfake_probability")
    if deepfake_prob is not None:
        vd_confidence = "high" if deepfake_prob > 0.3 or deepfake_prob < 0.1 else "low"
        visual_df_payload = {
            "module": "visual_deepfake",
            "session_id": session_id,
            "timestamp": now_ts,
            "deepfake_probability": round(float(deepfake_prob), 4),
            "confidence": vd_confidence,
            "cascade_stage": 1,
            "frames_scored_last_30s": 30,
            "latency_ms_avg": 12,
            "skipped": False
        }
        state.latest_scores["visual_deepfake"] = visual_df_payload
        state.last_seen["visual_deepfake"] = time.time()
        if deepfake_prob > 0.7:
            log_event(state, "visual_deepfake_alert",
                       "Visual deepfake probability elevated ({}%, high confidence)".format(
                           round(deepfake_prob * 100, 1)))

    # Audio: use audio_spoof_probability from signals
    audio_prob = signals.get("audio_spoof_probability")
    if audio_prob is not None:
        audio_score = round((1.0 - float(audio_prob)) * 100)
        audio_verdict = "FAKE" if audio_prob > 0.5 else "REAL"
        audio_payload = {
            "module": "audio",
            "session_id": session_id,
            "timestamp": now_ts,
            "voice_detected": True,
            "audio_deepfake_score": max(0, min(100, audio_score)),
            "verdict": audio_verdict
        }
        state.latest_scores["audio"] = audio_payload
        state.last_seen["audio"] = time.time()
        if audio_verdict == "FAKE":
            log_event(state, "audio_alert", "Audio deepfake signal flagged")

    # Security / injection: use injection_risk_score + virtual_camera_detected
    # from signals, now that main.py actually wires in security_module (fix #3).
    injection_risk_score = signals.get("injection_risk_score")
    virtual_camera_detected = signals.get("virtual_camera_detected")
    if injection_risk_score is not None:
        security_payload = {
            "module": "security",
            "session_id": session_id,
            "timestamp": now_ts,
            "virtual_camera_detected": bool(virtual_camera_detected),
            "device_name": signals.get("device_name") or "reported via fusion",
            "injection_risk_score": max(0, min(100, round(injection_risk_score))),
            "frame_timing_anomaly_score": max(0, min(100, round(100 - injection_risk_score))),
            "verdict": "FAKE" if virtual_camera_detected or injection_risk_score < 50 else "REAL"
        }
        state.latest_scores["security"] = security_payload
        state.last_seen["security"] = time.time()
        if virtual_camera_detected:
            log_event(state, "injection_alert", "Virtual camera / injection signature detected")

    # Hard-deny reasons (fix #1 + #2): these now actually override the
    # weighted-average band in compute_trust(), not just get logged.
    hard_deny = payload.get("hard_deny_reasons", [])
    state._latest_hard_deny_reasons = hard_deny
    state._hard_deny_last_seen = time.time() if hard_deny else state._hard_deny_last_seen
    if hard_deny:
        log_event(state, "fusion_verdict",
                   "Access denied - {}".format(", ".join(hard_deny).replace("_", " ")))


@app.route('/fusion', methods=['POST'])
def receive_fusion():
    """Thin HTTP wrapper around ingest_fusion() for main.py's local mode."""
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "empty or invalid JSON body"}), 400
    _room, state = resolve_state(
        payload.get("room_code") or request.args.get("code"),
        payload.get("session_id") or LOCAL_SESSION, create=True)
    ingest_fusion(payload, state)
    return jsonify({"received": True, "source": "fusion", "signals_mapped": True})


@app.route('/scores', methods=['GET'])
def get_scores():
    """
    Returns all latest module signals plus a fused trust score computed
    server-side. Uses Week 3 weights, EMA smoothing, conflict detection,
    hard-deny override, and missing-signal handling. This is what the
    dashboard polls, and what main.py's on-screen overlay now also polls
    (see fix #1) so the two never show a different number.

    Scoped: /scores?code=K7M2Q9[&session=web_ab12]. Without a code it serves
    the LOCAL room (main.py). Without a session it serves the room's most
    recently active client, and always returns the room's client roster so
    the admin can switch focus.
    """
    code = request.args.get("code")
    session_id = request.args.get("session")
    room = get_room(code)
    if room is None:
        return jsonify({"error": "unknown room code", "code": code}), 404
    prune_rooms()
    state = (room.client(session_id) if session_id
             else (room.most_recent_client() or room.client(LOCAL_SESSION)))

    # Flatten all scores into one dict for fusion
    flat_scores = {}
    verdicts = []
    module_payloads = {}  # keep payloads for trust-contribution lookups

    v = state.latest_scores.get("vision")
    vision_inner = None
    if v:
        vision_inner = v.get("payload")
        flat_scores["liveness_score"] = derive_liveness_score(vision_inner)
        verdicts.append("FAKE" if vision_inner and not vision_inner.get("challenge_passed") else "REAL")
        module_payloads["vision"] = v

    # M1's visual_deepfake classifier: separate module, deepfake_probability
    # is 0.0-1.0 where higher = more fake. Convert to the 0-100
    # higher-is-more-trustworthy scale everything else in this file uses.
    vd = state.latest_scores.get("visual_deepfake")
    if vd:
        deepfake_probability = vd.get("deepfake_probability")
        if deepfake_probability is not None:
            flat_scores["visual_deepfake_score"] = round((1.0 - deepfake_probability) * 100)
        verdicts.append("FAKE" if deepfake_probability is not None and deepfake_probability > 0.5 else "REAL")
        module_payloads["visual_deepfake"] = vd

    # Pass through optional M2 fields for dashboard
    # (rtf is available in the raw payload for the frontend)
    a = state.latest_scores.get("audio")
    if a:
        flat_scores["audio_deepfake_score"] = a.get("audio_deepfake_score")
        verdicts.append(a.get("verdict", "REAL"))
        module_payloads["audio"] = a

    s = state.latest_scores.get("security")
    if s:
        flat_scores["injection_risk_score"]      = s.get("injection_risk_score")
        flat_scores["frame_timing_anomaly_score"] = s.get("frame_timing_anomaly_score")
        verdicts.append(s.get("verdict", "REAL"))
        module_payloads["security"] = s

    i = state.latest_scores.get("identity")
    identity_detail = None
    if i:
        flat_scores["identity_score"] = i.get("identity_score")
        verdicts.append(i.get("verdict", "REAL"))
        identity_detail = {
            "identity_score":           i.get("identity_score"),
            "similarity_score":         i.get("similarity_score"),
            "threshold_used":           i.get("threshold_used"),
            "confidence":               i.get("confidence"),
            "face_detected":            i.get("face_detected"),
            "multiple_faces_detected":  i.get("multiple_faces_detected"),
            "verdict":                  i.get("verdict"),
            "timestamp":                i.get("timestamp")
        }
        module_payloads["identity"] = i

    # Compute trust contributions (sub-scores) for each signal.
    # Missing signals get a neutral 0.5 placeholder.
    sub_scores = {}
    sub_scores["identity_score"] = normalize_to_trust_contribution(
        "identity_score", flat_scores.get("identity_score"), module_payloads.get("identity")
    )
    sub_scores["liveness_score"] = normalize_to_trust_contribution(
        "liveness_score", flat_scores.get("liveness_score"), module_payloads.get("vision")
    )
    sub_scores["visual_deepfake_score"] = normalize_to_trust_contribution(
        "visual_deepfake_score", flat_scores.get("visual_deepfake_score"), module_payloads.get("visual_deepfake")
    )
    sub_scores["audio_deepfake_score"] = normalize_to_trust_contribution(
        "audio_deepfake_score", flat_scores.get("audio_deepfake_score"), module_payloads.get("audio")
    )
    sub_scores["injection_risk_score"] = normalize_to_trust_contribution(
        "injection_risk_score", flat_scores.get("injection_risk_score"), module_payloads.get("security")
    )

    # Conflict detection MUST run before compute_trust() now, since the
    # conflict penalty is applied inside compute_trust().
    conflicts = detect_conflicts(flat_scores)

    raw_trust, band, hard_deny_reasons, identity_capped, conflict_penalty_pct = compute_trust(
        state,
        flat_scores, sub_scores, conflicts
    )
    ema_trust = apply_ema(state, raw_trust) if raw_trust is not None else None

    signal_missing = get_missing_signals(state)

    if band != "awaiting" and band != state._last_trust_band:
        log_event(state, "band_change",
                   "Trust band changed to \"{}\" (score: {})".format(band, raw_trust),
                   band=band)
    state._last_trust_band = band

    if band == "awaiting" or all(v is None for v in state.latest_scores.values()):
        reason = "Session starting — waiting for verification modules to report."
    else:
        reason = build_reason(
            flat_scores, verdicts,
            vision_inner.get("fail_reason") if vision_inner else None,
            identity_detail.get("confidence") if identity_detail else None,
            conflicts=conflicts,
            hard_deny_reasons=hard_deny_reasons,
            identity_capped=identity_capped,
            conflict_penalty_pct=conflict_penalty_pct
        )

    challenge_detail = None
    if vision_inner:
        challenge_detail = {
            "challenge_type":    vision_inner.get("challenge_type"),
            "prompt":            vision_inner.get("prompt"),
            "timeout_sec":       vision_inner.get("timeout_sec"),
            "challenge_passed":  vision_inner.get("challenge_passed"),
            "fail_reason":       vision_inner.get("fail_reason"),
            "response_time_ms":  vision_inner.get("metrics", {}).get("response_time_ms"),
            "timestamp":         v.get("timestamp") if v else None
        }

    visual_deepfake_detail = None
    if vd:
        visual_deepfake_detail = {
            "deepfake_probability":     vd.get("deepfake_probability"),
            "confidence":               vd.get("confidence"),
            "cascade_stage":            vd.get("cascade_stage"),
            "frames_scored_last_30s":   vd.get("frames_scored_last_30s"),
            "latency_ms_avg":           vd.get("latency_ms_avg"),
            "skipped":                  vd.get("skipped"),
            "timestamp":                vd.get("timestamp")
        }

    return jsonify({
        "trust_score": raw_trust,
        "ema_trust_score": ema_trust,
        "trust_band":  band,
        "reason":      reason,
        "conflict_detected": conflicts.get("conflict_detected", False),
        "conflict_types": conflicts.get("conflict_types", []),
	"conflict_penalty_applied": conflict_penalty_pct,
        "identity_capped": identity_capped,
        "hard_deny_reasons": hard_deny_reasons,
        "challenge":   challenge_detail,
        "identity":    identity_detail,
        "visual_deepfake": visual_deepfake_detail,
        "challenge_history": [
            {
                "challenge_type":   ev["payload"].get("challenge_type"),
                "prompt":           ev["payload"].get("prompt"),
                "challenge_passed": ev["payload"].get("challenge_passed"),
                "fail_reason":      ev["payload"].get("fail_reason"),
                "response_time_ms": ev["payload"].get("metrics", {}).get("response_time_ms"),
                "timestamp":        ev.get("timestamp")
            }
            for ev in state.vision_challenge_history[-10:]
        ],
        "signals": {
            "identity_score":          flat_scores.get("identity_score"),
            "liveness_score":          flat_scores.get("liveness_score"),
            "visual_deepfake_score":   flat_scores.get("visual_deepfake_score"),
            "audio_deepfake_score":    flat_scores.get("audio_deepfake_score"),
            "injection_risk_score":    flat_scores.get("injection_risk_score")
        },
        "sub_scores": {
            "identity_trust":        sub_scores.get("identity_score"),
            "liveness_trust":        sub_scores.get("liveness_score"),
            "visual_trust":          sub_scores.get("visual_deepfake_score"),
            "audio_trust":           sub_scores.get("audio_deepfake_score"),
            "injection_trust":       sub_scores.get("injection_risk_score")
        },
        "tiers": {
            "identity":       score_to_tier(flat_scores["identity_score"])        if flat_scores.get("identity_score")        is not None else None,
            "liveness":       score_to_tier(flat_scores["liveness_score"])        if flat_scores.get("liveness_score")        is not None else None,
            "visual_deepfake":score_to_tier(flat_scores["visual_deepfake_score"]) if flat_scores.get("visual_deepfake_score") is not None else None,
            "audio_deepfake": score_to_tier(flat_scores["audio_deepfake_score"])  if flat_scores.get("audio_deepfake_score")  is not None else None,
            "injection_risk": score_to_tier(flat_scores["injection_risk_score"])  if flat_scores.get("injection_risk_score")  is not None else None
        },
        "signal_missing": signal_missing,
        "camera": {
            "name":  state.camera_name,
            "class": state.camera_class,   # "physical" | "suspect" | "virtual" | "unknown" | None
        },
        "modules_reporting": [k for k, v in state.latest_scores.items() if v is not None],
        "session_log": state.session_log[-30:],
        # Admin view: who else is verifying in this room right now.
        "room_code": room.code,
        "focused_session": state.session_id,
        "clients": [
            {
                "session_id": c.session_id,
                "trust_score": c.summary.get("trust_score"),
                "band": c.summary.get("band"),
                "verdict": c.summary.get("verdict"),
                "session_final": c.summary.get("session_final", False),
                "idle_sec": round(time.time() - c.last_active, 1),
            }
            for c in sorted(room.clients.values(),
                            key=lambda c: -c.last_active)
            if c.session_id != LOCAL_SESSION or room.code == DEFAULT_CODE
        ],
        "raw": state.latest_scores
    })


@app.route('/session-log', methods=['GET'])
def get_session_log():
    """Full session log, for a dedicated audit-trail view if the dashboard needs more than the last 30 entries embedded in /scores."""
    _room, state = resolve_state(request.args.get("code"),
                                 request.args.get("session"))
    if state is None:
        return jsonify({"error": "unknown room code"}), 404
    return jsonify({"log": state.session_log, "count": len(state.session_log)})


@app.route('/session-reset', methods=['POST'])
def reset_session():
    """
    Clears all module state and the session log. Used by the dashboard's
    "session reset" button so multiple demo scenarios can be run
    back-to-back without restarting the whole backend.
    """
    code = request.args.get("code")
    err = _coded_room_write_blocked(code)
    if err:
        return jsonify({"error": err}), 401
    _room, state = resolve_state(code, request.args.get("session"),
                                 create=(not code or str(code).upper() == DEFAULT_CODE))
    if state is None:
        return jsonify({"error": "unknown room code"}), 404
    for key in state.latest_scores:
        state.latest_scores[key] = None
    for key in state.last_seen:
        state.last_seen[key] = None
    state.vision_challenge_history.clear()
    state.session_log.clear()
    state._last_trust_band = None
    state._ema_trust_score = None
    state._latest_hard_deny_reasons = []
    state._hard_deny_last_seen = None
    log_event(state, "session_reset", "Session was reset.")
    return jsonify({"reset": True})


# ─────────────────────────────────────────────
# WebRTC signaling (unchanged from before)
# ─────────────────────────────────────────────

# room -> {'verifier': sid or None, 'participant': sid or None}
# Tracking by ROLE instead of join order makes offerer-selection
# deterministic: it no longer matters which page loads/joins first,
# and it also correctly re-fires if one side refreshes mid-session.
_rooms = {}

# socket sid -> (session_id, room_code). Filled by on_analysis_frame so
# on_disconnect can free the engine slot IMMEDIATELY when a participant
# closes the tab / refreshes, instead of holding one of the
# MAX_ACTIVE_SESSIONS slots hostage for STALE_AFTER_SEC (60s) - which is
# exactly long enough to show "Server at capacity" to the next person
# during a live demo.
_SOCKET_SESSIONS = {}

@socketio.on('join')
def on_join(data):
    room = data.get('room')
    role = data.get('role')  # 'verifier' or 'participant'
    if not room or role not in ('verifier', 'participant'):
        return

    join_room(room)
    state = _rooms.setdefault(room, {'verifier': None, 'participant': None})
    state[role] = request.sid

    emit('peer-joined', {'role': role}, room=room, include_self=False)

    # Fire only once both roles are present -- safe to check on EVERY
    # join, since whichever join completes the pair is the one that
    # should trigger it.
    if state['verifier'] and state['participant']:
        emit('ready-to-offer', {}, room=state['verifier'])

@socketio.on('disconnect')
def on_disconnect():
    for state in _rooms.values():
        for role in ('verifier', 'participant'):
            if state.get(role) == request.sid:
                state[role] = None
    # Free this participant's engine slot right away (see _SOCKET_SESSIONS
    # comment). Without this, only an explicit 'analysis_end' releases the
    # slot, and a tab-close/refresh leaks it for STALE_AFTER_SEC.
    mapped = _SOCKET_SESSIONS.pop(request.sid, None)
    if mapped:
        sess, code = mapped
        drop_session(sess)
        room = get_room(code)
        if room is not None:
            room.clients.pop(sess, None)
        print(f"[ROOM {code}] client disconnected: {sess} (engine slot freed)")

@socketio.on('offer')
def on_offer(data):
    emit('offer', data, room=data['room'], include_self=False)
    
@socketio.on('answer')
def on_answer(data):
    emit('answer', data, room=data['room'], include_self=False)

@socketio.on('ice-candidate')
def on_ice_candidate(data):
    emit('ice-candidate', data, room=data['room'], include_self=False)


# ─────────────────────────────────────────────
# Faza 1: browser frame ingestion → AnalysisEngine
# participant.html emits 'analysis_frame' at ~5fps with a base64 JPEG.
# The engine drops frames if it's still busy, so a slow inference step
# can never queue up and snowball latency.
# ─────────────────────────────────────────────
@socketio.on('analysis_frame')
def on_analysis_frame(data):
    sid = data.get('session_id')
    b64 = data.get('jpg', '')
    if not sid or ',' not in b64:
        return
    try:
        raw = base64.b64decode(b64.split(',', 1)[1])
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return
    if frame is None:
        return

    code = (data.get('code') or DEFAULT_CODE).upper()
    room = get_room(code)
    if room is None:
        emit('room_closed', {'message': 'Session code not found or expired'})
        return

    eng = get_engine(sid, remote=True)
    if eng is None:
        emit('analysis_busy',
             {'message': 'Server at capacity - please wait a moment and retry'})
        return

    ear = data.get('ear')
    yaw_ratio = data.get('yaw_ratio')
    _SOCKET_SESSIONS[request.sid] = (sid, code)   # for on_disconnect cleanup

    fusion = eng.process_frame(
        frame,
        ear=ear if isinstance(ear, (int, float)) else None,
        yaw_ratio=yaw_ratio if isinstance(yaw_ratio, (int, float)) else None)
    if fusion:
        state = room.client(sid)          # per-client state, not a global one

        # ── Camera integrity (server-authoritative) ─────────────────
        # participant.html sends the MediaStreamTrack label with every
        # frame. The browser's own HARD/SOFT verdict is display-only;
        # the server re-classifies the NAME here via
        # security/device_check.py, so one server-side list decides what
        # counts as an injection tool. NOTE: the label is client-supplied
        # and a modified client can lie about it - this is a deterrent
        # against off-the-shelf tools (OBS etc.), not a cryptographic
        # guarantee. Document as such in the threat model.
        cam_name = data.get('camera_name')
        cam_class, _cam_match = classify_camera_label(cam_name)
        if cam_name and cam_name != state.camera_name:
            state.camera_name = cam_name
            state.camera_class = cam_class
            print(f"[ROOM {room.code}] {sid} camera: {cam_name} ({cam_class})")
            log_event(state, "camera_info",
                      "Active camera: {} [{}]".format(cam_name, cam_class.upper()))

        # Wire the classification into the fusion payload BEFORE
        # ingest_fusion() so state, the /scores fusion, the dashboard and
        # the participant all see one consistent verdict.
        #   HARD ('virtual')  -> hard deny + fraud (mirrors main.py local path)
        #   SOFT ('suspect')  -> 15% score penalty, NEVER a denial
        #   'physical'        -> healthy injection signal (90)
        #   'unknown'         -> injection signal stays MISSING, as before
        sigs = fusion.setdefault('signals', {})
        if cam_class == 'virtual':
            deny = fusion.setdefault('hard_deny_reasons', [])
            if 'virtual_camera_detected' not in deny:
                deny.append('virtual_camera_detected')
            sigs['virtual_camera_detected'] = True
            sigs['injection_risk_score'] = 5
            sigs['device_name'] = cam_name
            fusion['trust_score'] = min(fusion.get('trust_score', 50), 15)
            fusion['trust_band'] = 'fraud'
            fusion['verdict'] = 'ACCESS DENIED'
        elif cam_class == 'suspect':
            sigs['virtual_camera_detected'] = False
            sigs['injection_risk_score'] = 55
            sigs['device_name'] = cam_name
            penalized = int(round(fusion.get('trust_score', 50) * 0.85))
            fusion['trust_score'] = penalized
            # Re-derive band with the same 80/50 thresholds engine.py uses.
            fusion['trust_band'] = ('trusted' if penalized >= 80
                                    else 'suspicious' if penalized >= 50
                                    else 'fraud')
            if not fusion.get('hard_deny_reasons'):
                fusion['verdict'] = ('ACCESS GRANTED'
                                     if fusion['trust_band'] in ('trusted', 'suspicious')
                                     else 'ACCESS DENIED')
        elif cam_class == 'physical':
            sigs.setdefault('virtual_camera_detected', False)
            sigs['injection_risk_score'] = 90
            sigs['device_name'] = cam_name
        # ─────────────────────────────────────────────────────────────

        ingest_fusion(fusion, state)
        state.summary = {
            "trust_score": fusion.get("trust_score"),
            "band": fusion.get("trust_band"),
            "verdict": fusion.get("verdict"),
            "session_final": fusion.get("session_final", False),
        }
        emit('verdict_update', {
            'session_id': sid,
            'trust_score': fusion['trust_score'],
            'trust_band': fusion['trust_band'],
            'verdict': fusion['verdict'],
            'hard_deny_reasons': fusion['hard_deny_reasons'],
            'active_challenge': fusion.get('active_challenge'),
            'challenge_result': fusion.get('challenge_result'),
            # Voice (2nd factor) fields - previously computed in engine.py's
            # fusion payload but never forwarded here, so participant.html
            # never received them no matter how correct the backend logs
            # looked. Bug found by comparing backend logs (NEW VOICE
            # CHALLENGE fired) against browser console (nothing arrived).
            'active_audio_challenge': fusion.get('active_audio_challenge'),
            'audio_challenge_result': fusion.get('audio_challenge_result'),
            'factors': fusion.get('factors'),
            'session_final': fusion.get('session_final', False),
        })


@socketio.on('analysis_end')
def on_analysis_end(data):
    sid = (data or {}).get('session_id')
    code = (data or {}).get('code')
    if sid:
        drop_session(sid)
        room = get_room(code)
        if room is not None:
            room.clients.pop(sid, None)
            print(f"[ROOM {room.code}] client left: {sid}")


@socketio.on('audio_clip')
def on_audio_clip(data):
    """Voice-challenge clip from participant.html: raw little-endian
    float32 PCM (base64) + its sample_rate. Resampled to 16k here (simple
    linear interp - fine for speech) and handed to the session engine,
    which runs phrase verification + AASIST spoof scoring on it."""
    sid = (data or {}).get('session_id')
    b64 = (data or {}).get('pcm_b64', '')
    sr = int((data or {}).get('sample_rate', 48000))
    if not sid or not b64:
        return
    fmt = (data or {}).get('format', 'float32')
    try:
        raw = base64.b64decode(b64)
        if fmt == 'int16':
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            pcm = np.frombuffer(raw, dtype=np.float32)
    except Exception as e:
        print(f"[AUDIO] could not decode clip: {e}")
        return
    print(f"[AUDIO] clip received: session={sid} {pcm.size/sr:.1f}s @ {sr}Hz "
          f"peak={float(np.abs(pcm).max()):.3f}")
    if pcm.size < sr // 4:   # <250ms is useless
        return
    if sr != 16000:
        target_len = int(len(pcm) * 16000 / sr)
        pcm = np.interp(np.linspace(0, len(pcm) - 1, target_len),
                        np.arange(len(pcm)), pcm).astype(np.float32)
    room = get_room((data or {}).get('code'))
    if room is not None:
        room.client(sid)                  # refresh liveness of this client
    eng = get_engine(sid, remote=True)
    if eng is None:
        return
    eng.process_audio_clip(pcm)


# ─────────────────────────────────────────────
# Faza 1: serve the frontend from Flask so participant/dashboard run on
# the same origin as the API (no file://, no CORS surprises, and
# localhost counts as a secure context for getUserMedia).
#   http://localhost:5000/participant  → participant.html
#   http://localhost:5000/dashboard    → index.html
# ─────────────────────────────────────────────
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))


@app.route('/api/admin/status', methods=['GET'])
def admin_status():
    """Tells the UI whether this deployment already has a PIN, so the first
    admin sees "Create a PIN" and everyone after sees "Sign in"."""
    return jsonify({"pin_set": admin_pin() is not None})


@app.route('/api/admin/setup', methods=['POST'])
def admin_setup():
    """First-run PIN creation. Refuses once a PIN exists, so a second person
    can't silently overwrite the first admin's PIN - they must enter it."""
    with _pin_lock:
        if admin_pin() is not None:
            return jsonify({"error": "a PIN is already set for this server"}), 409
        pin = str((request.get_json(silent=True) or {}).get("pin", "")).strip()
        if len(pin) < 4 or not pin.isdigit():
            return jsonify({"error": "PIN must be at least 4 digits"}), 400
        _save_pin(pin)
        print("[ADMIN] PIN created")
        return jsonify({"ok": True})


@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """Shared-PIN gate for the verifier console. Deliberately simple: this is
    a demo console, not a production IAM."""
    current = admin_pin()
    if current is None:
        return jsonify({"ok": False, "error": "no PIN set", "needs_setup": True}), 409
    pin = str((request.get_json(silent=True) or {}).get("pin", ""))
    if pin != current:
        return jsonify({"ok": False, "error": "invalid PIN"}), 401
    return jsonify({"ok": True})


@app.route('/api/session/create', methods=['POST'])
def create_session():
    pin = str((request.get_json(silent=True) or {}).get("pin", ""))
    if admin_pin() is None or pin != admin_pin():
        return jsonify({"error": "invalid PIN"}), 401
    prune_rooms()
    code = new_code()
    return jsonify({"code": code})


@app.route('/api/session/<code>', methods=['GET'])
def session_exists(code):
    room = get_room(code)
    if room is None:
        return jsonify({"exists": False}), 404
    return jsonify({
        "exists": True,
        "code": room.code,
        "clients": len(room.clients),
        "created_at": room.created_at,
    })


@app.route('/')
def landing_page():
    return send_from_directory(FRONTEND_DIR, 'landing.html')


@app.route('/admin')
def admin_page():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/about')
def about_page():
    return send_from_directory(FRONTEND_DIR, 'about.html')


@app.route('/contact')
def contact_page():
    return send_from_directory(FRONTEND_DIR, 'contact.html')


# ─────────────────────────────────────────────
# Contact form
#
# No SMTP credentials are committed to this repo, so by default a message is
# APPENDED TO DISK (contact_messages.jsonl) with any attachment saved beside
# it, and the sender gets a success response. If SMTP env vars are provided
# the same message is also emailed to CONTACT_EMAIL:
#     DFG_SMTP_HOST, DFG_SMTP_PORT, DFG_SMTP_USER, DFG_SMTP_PASS
# (For Gmail, DFG_SMTP_PASS must be an App Password, not the account
# password.) Delivery failure never loses the message - it's on disk first.
# ─────────────────────────────────────────────
CONTACT_EMAIL = "python3scapy@gmail.com"
CONTACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contact")
CONTACT_LOG = os.path.join(CONTACT_DIR, "contact_messages.jsonl")
CONTACT_MAX_BYTES = 5 * 1024 * 1024
CONTACT_ALLOWED_EXT = {"pdf", "doc", "docx", "txt", "jpg", "jpeg", "png"}


def _send_contact_email(record, attachment_path=None):
    host = os.environ.get("DFG_SMTP_HOST")
    if not host:
        return False, "smtp_not_configured"
    try:
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = "[DeepfakeGuard] {} - {}".format(
            record.get("topic", "general"),
            record.get("subject") or "(no subject)")
        msg["From"] = os.environ.get("DFG_SMTP_USER", CONTACT_EMAIL)
        msg["To"] = CONTACT_EMAIL
        msg["Reply-To"] = record["email"]
        msg.set_content(
            "From: {name} <{email}>\nCompany: {company}\nTopic: {topic}\n"
            "Received: {received_at}\n\n{message}\n".format(**record))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                msg.add_attachment(f.read(), maintype="application",
                                   subtype="octet-stream",
                                   filename=os.path.basename(attachment_path))

        with smtplib.SMTP(host, int(os.environ.get("DFG_SMTP_PORT", 587))) as smtp:
            smtp.starttls()
            smtp.login(os.environ["DFG_SMTP_USER"], os.environ["DFG_SMTP_PASS"])
            smtp.send_message(msg)
        return True, "sent"
    except Exception as e:
        print("[CONTACT] SMTP send failed: {}".format(e))
        return False, str(e)


@app.route('/api/contact', methods=['POST'])
def contact_submit():
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip()
    message = (request.form.get("message") or "").strip()

    if not name or not email or not message:
        return jsonify({"error": "name, email and message are required"}), 400
    if "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"error": "invalid email address"}), 400
    if len(message) > 20000:
        return jsonify({"error": "message is too long"}), 400

    os.makedirs(CONTACT_DIR, exist_ok=True)

    attachment_path = None
    f = request.files.get("attachment")
    if f and f.filename:
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in CONTACT_ALLOWED_EXT:
            return jsonify({"error": "unsupported attachment type .{}".format(ext)}), 400
        safe = secure_filename(f.filename)
        stamped = "{}_{}".format(int(time.time()), safe)
        attachment_path = os.path.join(CONTACT_DIR, stamped)
        f.save(attachment_path)
        if os.path.getsize(attachment_path) > CONTACT_MAX_BYTES:
            os.remove(attachment_path)
            return jsonify({"error": "attachment exceeds 5MB"}), 400

    record = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "email": email,
        "company": (request.form.get("company") or "").strip(),
        "topic": (request.form.get("topic") or "general").strip(),
        "subject": (request.form.get("subject") or "").strip(),
        "message": message,
        "attachment": os.path.basename(attachment_path) if attachment_path else None,
    }

    with open(CONTACT_LOG, "a", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")

    emailed, detail = _send_contact_email(record, attachment_path)
    print("[CONTACT] {} <{}> - {} (email: {})".format(
        name, email, record["subject"] or "(no subject)", detail))

    return jsonify({"ok": True, "emailed": emailed})


@app.route('/participant')
def participant_page():
    return send_from_directory(FRONTEND_DIR, 'participant.html')


@app.route('/dashboard')
def dashboard_page():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/frontend/<path:fname>')
def frontend_static(fname):
    return send_from_directory(FRONTEND_DIR, fname)


if __name__ == '__main__':
    # Preload all shared models (deepfake, audio, InsightFace, Whisper ASR)
    # in the background so the FIRST participant's challenges aren't hit by
    # cold-start model downloads/loads.
    import threading as _threading
    from engine import warm_up as _warm_up
    _threading.Thread(target=_warm_up, daemon=True,
                      name="model-warmup").start()

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
