# DeepfakeGuard

Real-time multimodal deepfake detection and biometric verification for live video calls.

Built as a response to the [Arup deepfake fraud incident](https://www.cnn.com/2024/05/16/tech/arup-deepfake-fraud-hong-kong-intl-hnk) — where an employee joined a video call with what appeared to be senior colleagues, all of them AI-generated, and authorised a multi-million dollar transfer. DeepfakeGuard asks a simple question about every participant in a call: **is this a real, live, correct human?**

---

## What it does

The system fuses **five independent signals** into a single explainable trust score (0–100):

| Signal | Weight | What it detects |
|---|---|---|
| **Identity** | 0.25 | Is this the enrolled person? (ArcFace embedding similarity / session continuity) |
| **Liveness** | 0.25 | Randomised blink + head-turn + spoken-phrase challenges |
| **Visual deepfake** | 0.20 | Per-frame deepfake classifier (SigLIP2, HuggingFace) |
| **Audio spoof** | 0.20 | Voice clone / replay detection (XGBoost + AASIST) |
| **Injection** | 0.10 | Virtual camera & frame-timing anomalies (OBS, ManyCam, etc.) |

On top of the weighted score sit **hard-deny rules** — a detected hard virtual camera or a failed liveness challenge overrides the average and forces a `FRAUD` verdict regardless of how good the other signals look.

Output bands: `TRUSTED` → `SUSPICIOUS` → `FRAUD`, always with a human-readable reason.

---

## Architecture

```
┌─────────────────┐                    ┌──────────────────────┐
│  participant/   │  getUserMedia      │   admin dashboard    │
│  browser        │  MediaStreamTrack  │   (PIN-gated)        │
│                 │  .label            │                      │
└────────┬────────┘                    └──────────▲───────────┘
         │  Socket.IO                             │ verdict_update
         │  analysis_frame (JPEG)                 │ live trust score
         │  audio_clip (16k int16)                │
         ▼                                        │
┌────────────────────────────────────────────────┴───────────┐
│  backend/app.py   — Flask-SocketIO, rooms, session state   │
│  ROOMS[code].clients[session_id] → ClientState             │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│  engine.py — headless AnalysisEngine (one per session)     │
│                                                            │
│  vision/     liveness (MediaPipe) + deepfake (SigLIP2)     │
│  audio_module/  spoof (XGBoost/AASIST) + phrase (Whisper)  │
│  security/   identity (ArcFace) + camera injection check   │
│                    ↓                                       │
│              fusion → trust_score, band, hard_deny_reasons │
└────────────────────────────────────────────────────────────┘
```

Heavy models are loaded **once** at module level and shared across sessions; only per-session state (challenge state machine, spoof history, reference embedding) lives on the engine instance.

---

## Repository layout

```
deepfakeguard/
├── backend/
│   └── app.py                  Flask-SocketIO server, rooms, WebRTC signalling
├── frontend/
│   ├── participant.html        Camera capture, liveness HUD, challenge UI
│   └── admin.html              Live dashboard: trust score, signals, session log
├── vision/
│   ├── liveness.py             MediaPipe face mesh, EAR blink, head-pose
│   └── deepfake_detector.py    SigLIP2 classifier (HF, auto-download)
├── audio_module/
│   ├── audio_spoof.py          XGBoost / AASIST spoof scoring
│   ├── speech_challenge.py     faster-whisper phrase verification
│   └── train_xgb.py            Training script (ASVspoof 2019 LA)
├── security/
│   ├── identity.py             ArcFace 1:1 matching
│   ├── device_check.py         Virtual camera classification (hard / soft)
│   └── fusion_engine.py        Reference fusion impl (parity-tested, not live)
├── engine.py                   Headless AnalysisEngine — the live pipeline
├── main.py                     Local camera loop (debug / presenter mode)
├── tests/
│   └── test_fusion_parity.py   32,769-case parity test: live path == reference
└── requirements.txt
```

---

## Quick start

### Requirements

- Python 3.10+ (tested on 3.13, Windows 11)
- `ffmpeg` on PATH (audio extraction for uploaded files)
- A webcam and a microphone

### Install

```bash
git clone https://github.com/python3scapy-cmyk/deepfakeguard.git
cd deepfakeguard

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

> First run downloads model weights from Hugging Face (SigLIP2 deepfake classifier, InsightFace ArcFace, faster-whisper). Budget a few hundred MB and an internet connection.

### Run the server

```bash
python backend/app.py
```

Then open:

| Role | URL |
|---|---|
| Verifier / admin | `http://localhost:5000/admin` (PIN-gated) |
| Participant | `http://localhost:5000/participant` (enter the session code) |

The admin creates a room, shares the 6-character session code, and the participant joins. Analysis starts automatically once the camera stream is live.

### Local camera mode (no browser)

```bash
python main.py
```

Runs the identical pipeline against a local `cv2.VideoCapture` — useful for debugging without the WebRTC layer.

> **Windows note:** if the camera fails to open, use `cv2.VideoCapture(0, cv2.CAP_DSHOW)` — the default MSMF backend is unreliable on some Windows 11 setups.

---

## Running over HTTPS (required for remote demos)

`getUserMedia` only works on `localhost` or HTTPS. To demo across machines, tunnel the local server:

```bash
cloudflared tunnel --url http://localhost:5000
```

The frontend uses a same-origin Socket.IO connection, so tunnelling works with no code changes.

---

## Tests

```bash
pytest tests/
```

`test_fusion_parity.py` sweeps a 32,769-point grid across all five signals and asserts that the live scoring path (`backend/app.py` → `compute_trust`) agrees exactly with the reference implementation in `security/fusion_engine.py`. This is the guard against the project's original structural flaw: three divergent fusion implementations existing at once.

---

## Design principles

- **Single source of truth for fusion.** The unit-tested engine is the one powering the live dashboard. Enforced by the parity test.
- **Never silently degrade.** No fallbacks to mock or random data. If a model can't load, the system fails loudly — a silent mock masks integration failure and turns a security product into theatre.
- **The browser is the only honest source of camera identity.** Server-side device enumeration reports the *server's* cameras, not the remote participant's. `MediaStreamTrack.label` from `getUserMedia` is the correct signal.
- **Soft vs. hard denial.** Not every virtual camera is fraud. NVIDIA Broadcast and Continuity Camera are legitimate; OBS and ManyCam in a verification session are not. See [`FALSE_POSITIVES.md`](FALSE_POSITIVES.md).
- **Explainability over accuracy theatre.** Every verdict ships with the reason and the per-signal contributions.

---

## Documentation

| Doc | Contents |
|---|---|
| [`INTEGRATION_CONTRACT.md`](INTEGRATION_CONTRACT.md) | Signal payload schemas between modules |
| [`WEIGHT_RATIONALE.md`](WEIGHT_RATIONALE.md) | Why each signal carries the weight it does |
| [`THREAT_MODEL.md`](THREAT_MODEL.md) | What we defend against — and what we don't |
| [`FALSE_POSITIVES.md`](FALSE_POSITIVES.md) | Known benign triggers and how they're handled |
| [`EVALUATION.md`](EVALUATION.md) | APCER / BPCER metrics from recorded sessions |
| [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | Demo runbook |

---

## Limitations

- Cannot detect hardware-level HDMI loop injection.
- Cannot detect driver-level video injection below the browser.
- Frame-timing analysis needs per-camera-model calibration; it is excluded entirely from remote browser sessions, where it is meaningless.
- Deepfake classifier accuracy degrades in poor lighting and at low resolution.
- Requires CPU headroom: three concurrent sessions is the current cap (`MAX_ACTIVE_SESSIONS`).

---

## Team

| Member | Area |
|---|---|
| M1 | Vision — MediaPipe liveness, deepfake classifier |
| M2 | Audio — spoof detection, lip-sync, phrase challenge |
| M3 | Security — injection detection, identity matching, fusion |
| M4 | UI / Backend — Flask, WebRTC signalling, session management, integration |

---

## License

MIT
