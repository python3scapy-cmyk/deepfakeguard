from security.device_check import list_cameras, is_virtual_camera
from security.timing_check import get_timing_anomaly_score

def get_security_signal(camera_index=0):
    """Check camera and return security info"""
    try:
        devices = list_cameras()
    except Exception as e:
        return {
            "virtual_camera_detected": False,
            "device_name": f"ERROR: {str(e)}",
            "frame_timing_anomaly_score": 0.0,
        }
    if not devices:
        return {
            "virtual_camera_detected": False,
            "device_name": "NO_CAMERA_FOUND",
            "frame_timing_anomaly_score": 0.0,
        }
    device_name = devices[camera_index]
    is_fake, matched = is_virtual_camera(device_name)
    timing_score = get_timing_anomaly_score(camera_index=camera_index)
    return {
        "virtual_camera_detected": is_fake,
        "device_name": device_name,
        "frame_timing_anomaly_score": timing_score,
    }

def compute_security_score(security_payload):
    """Convert to 0-1 score"""
    if security_payload.get("virtual_camera_detected"):
        return 0.1
    timing_score = security_payload.get("frame_timing_anomaly_score", 0.0)
    return round(1.0 - timing_score, 2)

# Test
if __name__ == "__main__":
    import json
    print("=== Test ===")
    result = get_security_signal(0)
    print(json.dumps(result, indent=2))
    print(f"Score: {compute_security_score(result)}")