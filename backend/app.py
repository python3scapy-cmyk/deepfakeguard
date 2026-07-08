from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
from datetime import datetime, timezone

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)  # allows cross-origin HTTP requests from the dashboard

# ─────────────────────────────────────────────
# In-memory store — one entry per module,
# always the latest payload received.
# ─────────────────────────────────────────────
latest_scores = {
    "vision":   None,
    "audio":    None,
    "security": None,
    "identity": None
}

# Rolling history of M1's liveness challenge events (most recent last).
# Kept separate from latest_scores since M1 sends one terminal payload
# per challenge, not a running score.
vision_challenge_history = []
MAX_CHALLENGE_HISTORY = 20

# ─────────────────────────────────────────────
# Session event log — the audit trail behind the
# dashboard's session-log view. Every /score POST
# that represents a meaningful event (not just a
# routine repeat) gets appended here. This is the
# NIST SP 800-63-4 "continuous evaluation" artefact.
# ─────────────────────────────────────────────
session_log = []
MAX_LOG_ENTRIES = 500
_last_trust_band = None


def log_event(event_type, detail, band=None):
    session_log.append({
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "event_type":  event_type,   # e.g. "liveness_challenge", "identity_check", "band_change"
        "detail":      detail,
        "band":        band
    })
    if len(session_log) > MAX_LOG_ENTRIES:
        session_log.pop(0)

REQUIRED_FIELDS = {
    "vision":   ["module", "session_id", "timestamp", "payload"],
    "audio":    ["module", "session_id", "timestamp", "voice_detected",
                 "audio_deepfake_score", "verdict"],
    "security": ["module", "session_id", "timestamp",
                 "virtual_camera_detected", "injection_risk_score",
                 "frame_timing_anomaly_score", "verdict"],
    "identity": ["module", "session_id", "timestamp", "face_detected",
                 "multiple_faces_detected", "identity_score",
                 "similarity_score", "threshold_used", "confidence", "verdict"]
}

VALID_CONFIDENCE_LEVELS = {"high", "low", "none"}

# Fields required inside vision's nested "payload" object, per M1's
# confirmed contract (challenge_type, prompt, timeout_sec,
# challenge_passed, fail_reason, metrics.response_time_ms).
VISION_PAYLOAD_FIELDS = [
    "challenge_type", "prompt", "timeout_sec",
    "challenge_passed", "fail_reason", "metrics"
]

VALID_CHALLENGE_TYPES = {"blink", "turn_left", "turn_right"}
VALID_FAIL_REASONS = {"timeout", "wrong_direction", "spoof_detected", "multiple_faces_detected"}

SCORE_FIELDS = {
    "vision":   [],  # vision is validated separately via VISION_PAYLOAD_FIELDS
    "audio":    ["audio_deepfake_score"],
    "security": ["injection_risk_score", "frame_timing_anomaly_score"],
    "identity": ["identity_score"]
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


def compute_trust(scores):
    """
    Fuses all available module scores into one
    0–100 trust score and a band label.
    Only uses modules that have reported so far.
    """
    weights = {
        "identity_score":        0.20,
        "liveness_score":        0.20,
        "visual_deepfake_score": 0.20,
        "audio_deepfake_score":  0.20,
        "injection_risk_score":  0.20
    }

    total_weight = 0.0
    weighted_sum = 0.0

    for field, weight in weights.items():
        if field in scores and scores[field] is not None:
            weighted_sum += scores[field] * weight
            total_weight += weight

    if total_weight == 0:
        return None, "awaiting"

    trust = round(weighted_sum / total_weight)

    if trust >= 80:
        band = "trusted"
    elif trust >= 50:
        band = "suspicious"
    else:
        band = "fraud"

    return trust, band


def build_reason(scores, verdicts, vision_fail_reason=None, identity_confidence=None):
    """
    Builds a human-readable reason string based on
    which signals are flagged, for the dashboard
    explainability area.
    """
    flags = []

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
        return "All signals within normal thresholds. Biometric match confirmed."

    return "Score reduced — " + "; ".join(flags) + "."


# ─────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules_reporting": [k for k, v in latest_scores.items() if v is not None]
    })


@app.route('/score', methods=['POST'])
def receive_score():
    payload = request.get_json(silent=True)

    if not payload:
        return jsonify({"error": "empty or invalid JSON body"}), 400

    module = payload.get("module")
    if module not in latest_scores:
        return jsonify({
            "error": "unknown module '{}', expected one of: vision, audio, security, identity".format(module)
        }), 400

    valid, err = validate_payload(payload, module)
    if not valid:
        return jsonify({"error": err, "module": module}), 400

    latest_scores[module] = payload

    if module == "vision":
        vision_challenge_history.append(payload)
        if len(vision_challenge_history) > MAX_CHALLENGE_HISTORY:
            vision_challenge_history.pop(0)
        inner = payload["payload"]
        print("[{}] received vision challenge — type: {}, passed: {}, reason: {}".format(
            datetime.now(timezone.utc).strftime("%H:%M:%S"),
            inner.get("challenge_type"),
            inner.get("challenge_passed"),
            inner.get("fail_reason")
        ))
        type_name = inner.get("challenge_type", "").replace("_", " ")
        if inner.get("challenge_passed"):
            log_event("liveness_challenge",
                       "Liveness challenge \"{}\" passed ({} ms)".format(
                           type_name, inner.get("metrics", {}).get("response_time_ms")))
        else:
            log_event("liveness_challenge",
                       "Liveness challenge \"{}\" FAILED — {}".format(
                           type_name, (inner.get("fail_reason") or "").replace("_", " ")))
    elif module == "identity":
        print("[{}] received identity payload — score: {}, confidence: {}, verdict: {}".format(
            datetime.now(timezone.utc).strftime("%H:%M:%S"),
            payload.get("identity_score"),
            payload.get("confidence"),
            payload.get("verdict")
        ))
        log_event("identity_check",
                   "Identity check — {}% similarity, confidence: {}, verdict: {}".format(
                       round(payload.get("similarity_score", 0) * 100, 1),
                       payload.get("confidence"),
                       payload.get("verdict")))
    else:
        print("[{}] received {} payload — verdict: {}".format(
            datetime.now(timezone.utc).strftime("%H:%M:%S"),
            module,
            payload.get("verdict")
        ))
        if module == "security" and payload.get("virtual_camera_detected"):
            log_event("injection_alert", "Virtual camera / injection signature detected")
        elif module == "audio" and payload.get("verdict") == "FAKE":
            log_event("audio_alert", "Audio deepfake signal flagged")

    return jsonify({"received": True, "module": module})


@app.route('/scores', methods=['GET'])
def get_scores():
    """
    Returns all latest module signals plus a
    fused trust score computed server-side.
    This is what the dashboard polls every second.
    """
    # Flatten all scores into one dict for fusion
    flat_scores = {}
    verdicts = []

    v = latest_scores.get("vision")
    vision_inner = None
    if v:
        vision_inner = v.get("payload")
        flat_scores["liveness_score"] = derive_liveness_score(vision_inner)
        # vision has no top-level "verdict" field in M1's contract;
        # treat a failed challenge as a FAKE-equivalent signal for the
        # existing verdict-based reasoning below.
        verdicts.append("FAKE" if vision_inner and not vision_inner.get("challenge_passed") else "REAL")

    a = latest_scores.get("audio")
    if a:
        flat_scores["audio_deepfake_score"] = a.get("audio_deepfake_score")
        verdicts.append(a.get("verdict", "REAL"))

    s = latest_scores.get("security")
    if s:
        flat_scores["injection_risk_score"]      = s.get("injection_risk_score")
        flat_scores["frame_timing_anomaly_score"] = s.get("frame_timing_anomaly_score")
        verdicts.append(s.get("verdict", "REAL"))

    i = latest_scores.get("identity")
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

    trust, band = compute_trust(flat_scores)

    global _last_trust_band
    if band != "awaiting" and band != _last_trust_band:
        log_event("band_change",
                   "Trust band changed to \"{}\" (score: {})".format(band, trust),
                   band=band)
    _last_trust_band = band

    if band == "awaiting":
        reason = "Session starting — waiting for verification modules to report."
    else:
        reason = build_reason(
            flat_scores, verdicts,
            vision_inner.get("fail_reason") if vision_inner else None,
            identity_detail.get("confidence") if identity_detail else None
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

    return jsonify({
        "trust_score": trust,
        "trust_band":  band,
        "reason":      reason,
        "challenge":   challenge_detail,
        "identity":    identity_detail,
        "challenge_history": [
            {
                "challenge_type":   ev["payload"].get("challenge_type"),
                "prompt":           ev["payload"].get("prompt"),
                "challenge_passed": ev["payload"].get("challenge_passed"),
                "fail_reason":      ev["payload"].get("fail_reason"),
                "response_time_ms": ev["payload"].get("metrics", {}).get("response_time_ms"),
                "timestamp":        ev.get("timestamp")
            }
            for ev in vision_challenge_history[-10:]
        ],
        "signals": {
            "identity_score":          flat_scores.get("identity_score"),
            "liveness_score":          flat_scores.get("liveness_score"),
            "visual_deepfake_score":   flat_scores.get("visual_deepfake_score"),
            "audio_deepfake_score":    flat_scores.get("audio_deepfake_score"),
            "injection_risk_score":    flat_scores.get("injection_risk_score")
        },
        "tiers": {
            "identity":       score_to_tier(flat_scores["identity_score"])        if flat_scores.get("identity_score")        is not None else None,
            "liveness":       score_to_tier(flat_scores["liveness_score"])        if flat_scores.get("liveness_score")        is not None else None,
            "visual_deepfake":score_to_tier(flat_scores["visual_deepfake_score"]) if flat_scores.get("visual_deepfake_score") is not None else None,
            "audio_deepfake": score_to_tier(flat_scores["audio_deepfake_score"])  if flat_scores.get("audio_deepfake_score")  is not None else None,
            "injection_risk": score_to_tier(flat_scores["injection_risk_score"])  if flat_scores.get("injection_risk_score")  is not None else None
        },
        "modules_reporting": [k for k, v in latest_scores.items() if v is not None],
        "session_log": session_log[-30:],
        "raw": latest_scores
    })


@app.route('/session-log', methods=['GET'])
def get_session_log():
    """Full session log, for a dedicated audit-trail view if the dashboard needs more than the last 30 entries embedded in /scores."""
    return jsonify({"log": session_log, "count": len(session_log)})


@app.route('/session-reset', methods=['POST'])
def reset_session():
    """
    Clears all module state and the session log. Used by the dashboard's
    "session reset" button so multiple demo scenarios can be run
    back-to-back without restarting the whole backend.
    """
    global _last_trust_band
    for key in latest_scores:
        latest_scores[key] = None
    vision_challenge_history.clear()
    session_log.clear()
    _last_trust_band = None
    log_event("session_reset", "Session was reset.")
    return jsonify({"reset": True})


# ─────────────────────────────────────────────
# WebRTC signaling (unchanged from before)
# ─────────────────────────────────────────────
@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)
    emit('peer-joined', {}, room=room, include_self=False)

@socketio.on('offer')
def on_offer(data):
    emit('offer', data, room=data['room'], include_self=False)

@socketio.on('answer')
def on_answer(data):
    emit('answer', data, room=data['room'], include_self=False)

@socketio.on('ice-candidate')
def on_ice_candidate(data):
    emit('ice-candidate', data, room=data['room'], include_self=False)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)