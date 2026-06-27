\# Security Module — Virtual Camera / Injection Detection



\## What it does

Detects whether the active camera feed is a real physical webcam or a

software-injected/virtual camera (e.g. OBS Virtual Camera, Snap Camera,

ManyCam). Combines two independent signals into one trust signal.



\## How it works

1\. \*\*Device identity check\*\* (`device\_check.py`) — enumerates connected

&#x20;  camera devices and flags known virtual-camera driver names.

2\. \*\*Frame-timing check\*\* (`timing\_check.py`) — measures the variance in

&#x20;  time between frames. Real cameras show natural jitter (auto-exposure,

&#x20;  USB transfer delay); software-rendered feeds tend to be unnaturally

&#x20;  uniform. Variance is normalized to a 0–1 anomaly score.

3\. \*\*Fusion\*\* (`security\_module.py`) — combines both into one payload:

```json

&#x20;  {

&#x20;    "virtual\_camera\_detected": false,

&#x20;    "device\_name": "Integrated Camera",

&#x20;    "frame\_timing\_anomaly\_score": 0.0

&#x20;  }

```

&#x20;  `compute\_security\_score()` converts this into a single 0–1 score used

&#x20;  by `main.py`'s overall trust score.



\## How to run standalone



python -m security.security\_module



\## Test results (Week 1)

| Source              | virtual\_camera\_detected | frame\_timing\_anomaly\_score | Score |

|----------------------|--------------------------|------------------------------|-------|

| Real webcam          | false                    | 0.0                          | 1.0   |

| OBS Virtual Camera   | true                     | 1.0                          | 0.1   |



\## Known limitations

\- Frame-timing detection is exploratory — only tested against OBS, not

&#x20; other virtual-camera tools yet.

\- Device-name matching only catches tools on the known-names list; a

&#x20; renamed or unlisted virtual camera would bypass that check (timing

&#x20; signal is the backup for this case).

\- Legitimate streamers/educators sometimes use OBS for non-malicious

&#x20; reasons — this signal should weight the overall trust score, not

&#x20; unilaterally block access on its own.

