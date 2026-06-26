# ============================================
# YOUR JOB: Check if camera is REAL or FAKE
# ============================================

import platform
import subprocess

# List of known FAKE camera names (from your research)
KNOWN_VIRTUAL_CAMERA_NAMES = [
    "OBS Virtual Camera",
    "Snap Camera",
    "ManyCam",
    "Iriun Webcam",
    "EpocCam",
    "DroidCam",
    "NDI Webcam",
    "XSplit VCam",
]


def list_cameras():
    """Get all camera names on your computer"""
    system = platform.system()

    if system == "Windows":
        return _list_cameras_windows()
    elif system == "Darwin":  # Mac
        return _list_cameras_mac()
    else:  # Linux
        return _list_cameras_linux()


def _list_cameras_windows():
    """Windows: use pygrabber library"""
    from pygrabber.dshow_graph import FilterGraph
    graph = FilterGraph()
    return graph.get_input_devices()


def _list_cameras_mac():
    """Mac: use ffmpeg command"""
    result = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        stderr=subprocess.PIPE, text=True
    )
    lines = result.stderr.splitlines()
    devices = []
    for line in lines:
        if "] [" in line and "AVFoundation video devices" not in line:
            name = line.split("]")[-1].strip()
            if name:
                devices.append(name)
    return devices


def _list_cameras_linux():
    """Linux: use v4l2-ctl command"""
    result = subprocess.run(
        ["v4l2-ctl", "--list-devices"],
        stdout=subprocess.PIPE, text=True
    )
    lines = result.stdout.splitlines()
    devices = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("/dev"):
            devices.append(stripped)
    return devices


def is_virtual_camera(device_name):
    """Check if a camera name is on the fake list"""
    name_lower = device_name.lower()
    for flagged in KNOWN_VIRTUAL_CAMERA_NAMES:
        if flagged.lower() in name_lower:
            return True, flagged
    return False, None


# ============================================
# TEST: Run this to see if it works!
# ============================================
if __name__ == "__main__":
    print("=== Checking Your Cameras ===")
    
    cameras = list_cameras()
    print(f"\nFound {len(cameras)} camera(s):\n")
    
    for i, name in enumerate(cameras):
        is_fake, matched = is_virtual_camera(name)
        if is_fake:
            print(f"  [{i}] {name}  ❌ FAKE CAMERA!")
        else:
            print(f"  [{i}] {name}  ✅ Real camera")