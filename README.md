# DeepfakeGuard — Week 3 (consolidated, working build)

This is a clean rebuild of the whole project, consolidated so `main.py`
is the single entry point for both Member 1 (vision) and Member 2 (audio)
work. All the bugs from earlier sessions are fixed here:

- `NoneType has no attribute 'get'` on `challenge_result` — fixed with
  proper `challenge = liveness_payload.get("challenge_result")` + a
  truthiness check before ever calling `.get()` on it.
- `ambiguous truth value` on numpy face arrays (`if not faces:`) — fixed
  with a `_has_faces()` helper used everywhere instead.
- `np.corrcoef` "dimension mismatch" in lip-sync — fixed by always
  slicing both buffers to the *same* length before correlating.
- Camera only opening for menu option [1] and not [2] — fixed, both
  camera-based options now share one `_open_camera()` path.
- Haar cascade failing to download (`403 Forbidden` from GitHub, or no
  internet at all) — fixed by using the cascade XML that ships **inside**
  the `opencv-python` package itself (`cv2.data.haarcascades`), so no
  network access is needed at all. This was the root cause of "kamera
  açılmadı" / face-detection-never-loads in earlier sessions.

## Project layout

```
deepfakeguard/
├── main.py                       # single entry point — run this
├── requirements.txt
├── vision/
│   ├── face_detector.py          # face detection + 2D anti-spoof heuristics
│   ├── liveness_engine.py        # blink / head-turn challenge state machine
│   └── deepfake_detector.py      # cascaded visual deepfake classifier (Week 3)
└── audio_module/
    ├── lip_sync.py                # mouth/audio correlation proxy
    └── aasist_detector.py         # AASIST+XGBoost audio spoof ensemble (Week 3)
```

## Setup (Mac terminal)

```bash
cd ~/Downloads/deepfakeguard        # or wherever you unzipped it
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
python3 main.py
```

You'll be asked:
1. `[1] Camera` / `[2] Video file` / `[3] Auto-generated test video`
   — start with **[3]** to confirm everything runs with zero camera/mic
   permissions needed.
2. Whether to enable the mock audio pipeline too (`y`/`N`).

If you pick `[1]` and the camera doesn't open, macOS will tell you to go to
**System Settings → Privacy & Security → Camera** and enable Terminal
(or iTerm/VS Code, whichever you're running from), then fully quit
(Cmd+Q) and reopen the terminal app before trying again.

## What's real vs. mock right now

| Signal | Status |
|---|---|
| Face detection | ✅ Real (Haar cascade, bundled with OpenCV) |
| 2D anti-spoof (photo/screen detection) | ✅ Real heuristics (texture, moire, reflection, screen bezel, etc.) |
| Liveness challenge (blink / turn head) | ✅ Real, brightness/position-based |
| Visual deepfake probability | ✅ Real — SigLIP2-based classifier (`prithivMLmods/deepfake-detector-model-v1`, ~94% accuracy on its own eval set), auto-downloaded from Hugging Face on first run. Stage 1 (Laplacian texture) still runs first as a cheap filter; Stage 2 (real model) only runs when Stage 1 is ambiguous. |
| Audio spoof (AASIST+XGBoost) | 🟡 Mock — no real microphone capture wired in yet, uses synthetic random audio |
| Lip sync | 🟡 Correlation proxy against the same synthetic audio — real once real mic capture is added |

To go from mock to real audio, replace the `mock_audio = ...` / `audio_chunk = ...`
lines in `main.py`'s main loop with a real microphone stream (e.g. `sounddevice`
`InputStream`), and pass `model_path=` into `DeepfakeDetector(...)` once you have
real EfficientNet weights downloaded.

## Anti-spoof note (photo/screen detection)

The 2D anti-spoof heuristics (Laplacian texture, moire pattern, glass
reflection, screen bezel darkness, sharp edges) are inherently a
"best-effort" signal against phone/tablet screens — modern OLED/AMOLED
screens can look convincingly real to a 2D webcam. The **liveness
challenge** (blink twice / turn head) is the stronger signal here,
since a static photo genuinely cannot blink or turn on command. For
a production-grade anti-spoof system you'd eventually want a depth
sensor (TrueDepth/LiDAR) rather than relying on 2D heuristics alone —
worth stating explicitly in any pitch/demo as a known limitation.
