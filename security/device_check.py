# ============================================================
# M3: Camera integrity - two-tier virtual camera classification
# ============================================================
#
# TWO JOBS, TWO ENTRY POINTS:
#
#   1. classify_camera_label(label)  <- THE one used for remote participants.
#      backend/app.py calls this with the MediaStreamTrack label that
#      participant.html reads from getUserMedia and sends with every
#      analysis_frame. The BROWSER is the only place that can see the
#      remote participant's device name; this function is the server's
#      single source of truth for what that name means.
#
#   2. list_cameras() / is_virtual_camera()  <- LOCAL pipeline only
#      (main.py -> security_module). These enumerate the cameras of the
#      machine running the code, which is only meaningful when the
#      participant and the server are the same machine. NEVER use these
#      for remote sessions - that was the Week 4 mistake this file fixes.
#
# TWO TIERS (see FALSE_POSITIVES rationale):
#
#   HARD  -> tools whose primary purpose is feeding arbitrary video into
#            a "camera" (OBS, ManyCam, ...). Presence = injection vector.
#            Policy: hard deny + fraud verdict.
#   SOFT  -> legitimate enhancement/passthrough tools that register as a
#            virtual device but normally relay a real camera (NVIDIA
#            Broadcast, Continuity Camera, ...). Policy: trust-score
#            penalty, never a hard denial.
#
# KEEP IN SYNC: frontend/participant.html mirrors these lists for its
# display-only badge. The SERVER lists below are authoritative for the
# verdict - if you edit here, update participant.html's copy too (or
# better: serve these lists from an endpoint and delete the copy).

import platform
import subprocess

# Tier 1: HARD - injection tools. Any match = hard deny.
HARD_VIRTUAL_CAMERA_NAMES = [
    "OBS Virtual Camera",
    "OBS-Camera",
    "Snap Camera",
    "ManyCam",
    "Iriun Webcam",
    "Iriun",
    "EpocCam",
    "DroidCam",
    "iVCam",
    "NDI Webcam",
    "XSplit VCam",
    "SplitCam",
    "AlterCam",
    "FineCam",
    "Reincubate Camo",
    "e2eSoft",
    "Virtual Camera",
    "Virtual Webcam",
]

# Tier 2: SOFT - legitimate tools that present as virtual devices.
# Penalized, NEVER hard-denied (false-positive protection).
SOFT_VIRTUAL_CAMERA_NAMES = [
    "NVIDIA Broadcast",
    "Continuity Camera",
    "Logi Capture",
    "CyberLink YouCam",
    "mmhmm",
]

# Labels the browser sends when it could not read a real device name.
# These must classify as 'unknown', NOT 'physical' - an unreadable label
# is absence of evidence, not evidence of a physical camera.
_UNKNOWN_LABELS = {"", "unknown", "unknown device", "default"}


def classify_camera_label(label):
    """Classify a camera name string into a trust tier.

    Returns (cls, matched_name) where cls is one of:
        'virtual'  - matched the HARD list  -> hard deny
        'suspect'  - matched the SOFT list  -> score penalty only
        'physical' - a readable name that matched neither list
        'unknown'  - empty/unreadable label -> treat injection signal
                     as MISSING (don't reward, don't punish)

    matched_name is the list entry that fired, or None.
    Matching is case-insensitive substring, same as the browser side.
    """
    text = (label or "").strip()
    if text.lower() in _UNKNOWN_LABELS:
        return "unknown", None
    lowered = text.lower()
    for name in HARD_VIRTUAL_CAMERA_NAMES:
        if name.lower() in lowered:
            return "virtual", name
    for name in SOFT_VIRTUAL_CAMERA_NAMES:
        if name.lower() in lowered:
            return "suspect", name
    return "physical", None


# ============================================================
# LOCAL-machine enumeration (main.py pipeline ONLY - see header)
# ============================================================

def list_cameras():
    """Enumerate camera device names on THIS machine."""
    system = platform.system()
    if system == "Windows":
        return _list_cameras_windows()
    elif system == "Darwin":  # Mac
        return _list_cameras_mac()
    else:  # Linux
        return _list_cameras_linux()


def _list_cameras_windows():
    """Windows: pygrabber (DirectShow). Listed in requirements.txt with a
    win32 platform marker - if this import fails, pip install pygrabber."""
    from pygrabber.dshow_graph import FilterGraph
    graph = FilterGraph()
    return graph.get_input_devices()


def _list_cameras_mac():
    """Mac: ffmpeg avfoundation device listing."""
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
    """Linux: v4l2-ctl device listing."""
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
    """Backward-compatible wrapper kept for security_module.py.

    Returns (True, matched) ONLY for HARD-tier matches, so the local
    pipeline's hard-deny behavior is unchanged. SOFT-tier devices return
    (False, matched) - callers that care about the penalty tier should
    use classify_camera_label() instead.
    """
    cls, matched = classify_camera_label(device_name)
    return cls == "virtual", matched


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("=== classify_camera_label self-test ===")
    cases = [
        ("Integrated Camera (04f2:b6be)", "physical"),
        ("OBS Virtual Camera", "virtual"),
        ("obs-camera 4", "virtual"),
        ("NVIDIA Broadcast", "suspect"),
        ("iPhone (Continuity Camera)", "suspect"),
        ("", "unknown"),
        ("Unknown device", "unknown"),
        ("Logitech BRIO", "physical"),
        ("SplitCam Video Driver", "virtual"),
    ]
    failed = 0
    for label, expected in cases:
        cls, matched = classify_camera_label(label)
        mark = "OK " if cls == expected else "FAIL"
        if cls != expected:
            failed += 1
        print(f"  [{mark}] {label!r:42s} -> {cls:8s} (matched: {matched})")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")

    print("\n=== Local camera enumeration (this machine) ===")
    try:
        cameras = list_cameras()
        print(f"Found {len(cameras)} camera(s):")
        for i, name in enumerate(cameras):
            cls, matched = classify_camera_label(name)
            print(f"  [{i}] {name}  -> {cls.upper()}"
                  + (f" (matched: {matched})" if matched else ""))
    except Exception as e:
        print(f"(enumeration unavailable here: {e})")
