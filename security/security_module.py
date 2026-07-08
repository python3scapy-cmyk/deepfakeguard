import requests
import time
from security.device_check import list_cameras, is_virtual_camera
from security.timing_check import get_timing_anomaly_score
from datetime import datetime, timezone

def get_security_signal(camera_index=0, session_id="live_webrtc_call_xyz_123"):
    try:
        devices = list_cameras()
    except Exception as e:
        return {
            "module": "security",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "virtual_camera_detected": False,
            "device_name": f"ERROR: {str(e)}",
            "injection_risk_score": 100,
            "frame_timing_anomaly_score": 0,
            "verdict": "REAL"
        }
    if not devices:
        return {
            "module": "security",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "virtual_camera_detected": False,
            "device_name": "NO_CAMERA_FOUND",
            "injection_risk_score": 100,
            "frame_timing_anomaly_score": 0,
            "verdict": "REAL"
        }
    device_name = devices[camera_index]
    is_fake, matched = is_virtual_camera(device_name)
    timing_score_01 = get_timing_anomaly_score(camera_index=camera_index)
    injection_risk_score = round((1.0 - timing_score_01) * 100)
    frame_timing_anomaly_score = round(timing_score_01 * 100)
    verdict = "FAKE" if is_fake or timing_score_01 > 0.5 else "REAL"
    return {
        "module": "security",
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
