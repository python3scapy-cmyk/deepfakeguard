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
    "security": None
}

REQUIRED_FIELDS = {
    "vision":   ["module", "session_id", "timestamp", "face_detected",
                 "identity_score", "liveness_score",
                 "visual_deepfake_score", "verdict"],
    "audio":    ["module", "session_id", "timestamp", "voice_detected",
                 "audio_deepfake_score", "verdict"],
    "security": ["module", "session_id", "timestamp",
                 "virtual_camera_detected", "injection_risk_score",
                 "frame_timing_anomaly_score", "verdict"]
}

SCORE_FIELDS = {
    "vision":   ["identity_score", "liveness_score", "visual_deepfake_score"],
    "audio":    ["audio_deepfake_score"],
    "security": ["injection_risk_score", "frame_timing_anomaly_score"]
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

    for field in SCORE_FIELDS[module]:
        val = payload.get(field)
        if not isinstance(val, (int, float)):
            return False, "field '{}' must be a number".format(field)
        if not (0 <= val <= 100):
            return False, "field '{}' must be 0–100, got {}".format(field, val)

    if payload.get("verdict") not in ("REAL", "FAKE"):
        return False, "verdict must be 'REAL' or 'FAKE'"

    return True, None


def score_to_tier(score):
    """Maps a 0–100 score to a trust tier string."""
    if score >= 80:
        return "good"
    elif score >= 50:
        return "warn"
    return "bad"


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


def build_reason(scores, verdicts):
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
            flags.append("liveness challenge failed")
        elif scores["liveness_score"] < 80:
            flags.append("liveness confidence below threshold")

    if scores.get("identity_score") is not None:
        if scores["identity_score"] < 50:
            flags.append("identity mismatch detected")

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
            "error": "unknown module '{}', expected one of: vision, audio, security".format(module)
        }), 400

    valid, err = validate_payload(payload, module)
    if not valid:
        return jsonify({"error": err, "module": module}), 400

    latest_scores[module] = payload
    print("[{}] received {} payload — verdict: {}".format(
        datetime.now(timezone.utc).strftime("%H:%M:%S"),
        module,
        payload.get("verdict")
    ))

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
    if v:
        flat_scores["identity_score"]        = v.get("identity_score")
        flat_scores["liveness_score"]        = v.get("liveness_score")
        flat_scores["visual_deepfake_score"] = v.get("visual_deepfake_score")
        verdicts.append(v.get("verdict", "REAL"))

    a = latest_scores.get("audio")
    if a:
        flat_scores["audio_deepfake_score"] = a.get("audio_deepfake_score")
        verdicts.append(a.get("verdict", "REAL"))

    s = latest_scores.get("security")
    if s:
        flat_scores["injection_risk_score"]      = s.get("injection_risk_score")
        flat_scores["frame_timing_anomaly_score"] = s.get("frame_timing_anomaly_score")
        verdicts.append(s.get("verdict", "REAL"))

    trust, band = compute_trust(flat_scores)
    reason      = build_reason(flat_scores, verdicts)

    return jsonify({
        "trust_score": trust,
        "trust_band":  band,
        "reason":      reason,
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
        "raw": latest_scores
    })


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