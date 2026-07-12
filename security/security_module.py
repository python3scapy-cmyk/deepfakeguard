import requests
import time
from security.device_check import list_cameras, is_virtual_camera
from security.timing_check import get_timing_anomaly_score, compute_variance_score
from datetime import datetime, timezone


def _build_result(session_id, timestamp, status, virtual_camera_detected,
                   device_name, timing_score_01):
    if timing_score_01 is None:
        return {
            "module": "security",
            "session_id": session_id,
            "timestamp": timestamp,
            "status": status,
            "virtual_camera_detected": virtual_camera_detected,
            "device_name": device_name,
            "injection_risk_score": None,
            "frame_timing_anomaly_score": None,
            "verdict": "UNKNOWN"
        }
    injection_risk_score = round(timing_score_01 * 100)
    frame_timing_anomaly_score = round(timing_score_01 * 100)
    verdict = "FAKE" if virtual_camera_detected or timing_score_01 > 0.5 else "REAL"
    return {
        "module": "security",
        "session_id": session_id,
        "timestamp": timestamp,
        "status": "ok",
        "virtual_camera_detected": virtual_camera_detected,
        "device_name": device_name,
        "injection_risk_score": injection_risk_score,
        "frame_timing_anomaly_score": frame_timing_anomaly_score,
        "verdict": verdict
    }


def get_security_signal_from_deltas(deltas, camera_index=0, session_id="live_webrtc_call_xyz_123"):
    """Same signal as get_security_signal(), but the timing-anomaly half
    is computed from inter-frame deltas the CALLER already collected
    (main.py's SecurityMonitor feeds it timestamps from the main capture
    loop's own cap.read() calls) instead of opening a second
    cv2.VideoCapture(camera_index).

    This is the fix for the frame corruption/static seen on Windows: the
    old get_security_signal() opened its own capture on a background
    thread every ~8s while the main loop's capture was also reading the
    same physical device (often via a different backend, e.g. default vs
    DirectShow) -- two concurrent opens on one camera is exactly the kind
    of thing that produces garbled/torn frames. This path never touches
    the device at all; list_cameras()/is_virtual_camera() only enumerate
    device names (pygrabber/ffmpeg/v4l2-ctl), they don't open a capture.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        devices = list_cameras()
    except Exception as e:
        return _build_result(session_id, timestamp, "unavailable", None,
                              f"ERROR: {str(e)}", None)

    if not devices:
        return _build_result(session_id, timestamp, "unavailable", None,
                              "NO_CAMERA_FOUND", None)

    if camera_index >= len(devices):
        camera_index = 0
    device_name = devices[camera_index]
    is_fake, _matched = is_virtual_camera(device_name)

    timing_score_01 = compute_variance_score(deltas)
    return _build_result(session_id, timestamp,
                          "ok" if timing_score_01 is not None else "unavailable",
                          is_fake, device_name, timing_score_01)


def get_security_signal(camera_index=0, session_id="live_webrtc_call_xyz_123"):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        devices = list_cameras()

    except Exception as e:
        return {
            "module": "security",
            "session_id": session_id,
            "timestamp": timestamp,
            "status": "unavailable",
            "virtual_camera_detected": None,
            "device_name": f"ERROR: {str(e)}",
            "injection_risk_score": None,
            "frame_timing_anomaly_score": None,
            "verdict": "UNKNOWN"
        }


    if not devices:
        return {
            "module": "security",
            "session_id": session_id,
            "timestamp": timestamp,
            "status": "unavailable",
            "virtual_camera_detected": None,
            "device_name": "NO_CAMERA_FOUND",
            "injection_risk_score": None,
            "frame_timing_anomaly_score": None,
            "verdict": "UNKNOWN"
        }


    # protect index
    if camera_index >= len(devices):
        camera_index = 0


    device_name = devices[camera_index]


    is_fake, matched = is_virtual_camera(device_name)


    timing_score_01 = get_timing_anomaly_score(
        camera_index=camera_index
    )


    # timing check failed
    if timing_score_01 is None:
        return {
            "module": "security",
            "session_id": session_id,
            "timestamp": timestamp,
            "status": "unavailable",
            "virtual_camera_detected": is_fake,
            "device_name": device_name,
            "injection_risk_score": None,
            "frame_timing_anomaly_score": None,
            "verdict": "UNKNOWN"
        }


    injection_risk_score = round(
        timing_score_01 * 100
    )

    frame_timing_anomaly_score = round(
        timing_score_01 * 100
    )


    verdict = (
        "FAKE"
        if is_fake or timing_score_01 > 0.5
        else "REAL"
    )


    return {
        "module": "security",
        "session_id": session_id,
        "timestamp": timestamp,
        "status": "ok",
        "virtual_camera_detected": is_fake,
        "device_name": device_name,
        "injection_risk_score": injection_risk_score,
        "frame_timing_anomaly_score": frame_timing_anomaly_score,
        "verdict": verdict
    }

def post_security_signal(camera_index=0, session_id="live_webrtc_call_xyz_123"):
    payload = get_security_signal(camera_index, session_id)
    try:
        res = requests.post("http://localhost:5000/score", json=payload, timeout=3)
        print("Posted:", res.json())
    except Exception as e:
        print("Failed to post:", e)

if __name__ == "__main__":
    import json
    print("=== Test ===")
    result = get_security_signal(0)
    print(json.dumps(result, indent=2))
    while True:
        post_security_signal(0)
        time.sleep(2)
