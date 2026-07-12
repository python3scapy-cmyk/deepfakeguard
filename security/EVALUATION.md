\# Evaluation Report — M3



\## Status



\*\*Real system-level evaluation: NOT YET RUN.\*\* Section 1 below is a

template waiting for real numbers from `scripts/record\_session.py` +

`scripts/compute\_metrics.py` against the live stack (`backend/app.py` +

`main.py` + a real camera/mic). Do not fill it in by hand — regenerate

the table from `evaluation/sessions.csv` and paste the script's output.



Appendix A is \*\*synthetic-signal validation only\*\*. It confirms the

fusion arithmetic (weights, conflict penalty, identity cap, EMA) does

what the spec says when fed hand-picked numbers. It says nothing about

whether M1/M2/M3's real detectors actually produce those numbers against

a real attack. Do not cite Appendix A as evidence the system catches

real deepfakes — it is evidence the math is correct, full stop.



A previous draft of this file reported APCER 6.7% (1/15 wrong-person

attacks scored "Trusted"). That number is stale: it predates the

identity-threshold-hardening cap now in `security/fusion\_engine.py` /

`backend/app.py`'s `compute\_trust()`, which forces any session with

`identity\_trust < 0.60` to cap at 79/Suspicious regardless of every

other signal — see `tests/test\_fusion\_parity.py::test\_wrong\_person\_attack\_blocked`,

which pins this behavior as a regression test. Against current code, a

wrong-person attack cannot reach Trusted, so APCER for that scenario is

0.0%, not 6.7%. The 6.7% number has been removed from this file rather

than kept as a second, disagreeing copy.



`security/evaluation\_runner.py` has been deleted. It contained a second,

independent copy of this same report as a Python file that was not

valid Python (unescaped Markdown past the header line — it raised a

`SyntaxError` if anyone tried to run it). One evaluation report, one

file, one set of numbers.



\---



\## 1. Real System-Level Evaluation (Live Stack) — PENDING



\*\*Protocol\*\* (unchanged from the Week 3 roadmap): 15 bonafide + 15

attack sessions (3 per attack type), run against the full live stack,

recording the final band per session.



Steps:

1\. Terminal 1: `python backend/app.py`

2\. Terminal 2: `python main.py`

3\. Browser: verifier dashboard at `http://localhost:5000` (`frontend/index.html`)

4\. For each session:

&#x20;  - Reset session (dashboard button, or POST `/session-reset`)

&#x20;  - Start the scenario (sit down / hold up phone / start OBS / etc.)

&#x20;  - Run: `python scripts/record\_session.py --label <label> --duration 60`

&#x20;    (see `scripts/record\_session.py` for the label convention)

5\. After all 30 sessions: `python scripts/compute\_metrics.py evaluation/sessions.csv`



\*\*Decide the band-source convention before running any sessions\*\* and

use it for all 30: `record\_session.py` records both

`final\_band\_raw\_majority` (majority vote of the API's raw, unsmoothed

`trust\_band` over the tail of the recording) and `final\_band\_ema` (band

re-derived from the final EMA-smoothed score). These can disagree at

threshold crossings. Pick one, note which one in this file, and use

`--band-source` consistently in `compute\_metrics.py`.



\### Results



\*(paste `scripts/compute\_metrics.py` output here once the protocol is run)\*



| Metric | Value | Description |

|--------|-------|-------------|

| APCER | TBD | Attack sessions ending "Trusted" |

| BPCER | TBD | Bonafide sessions ending "Potential Fraud" |

| Band source used | TBD | `raw\_majority` or `ema` |



\### Band Distribution



| Category | Trusted | Suspicious | Potential Fraud |

|----------|---------|------------|-----------------|

| Bonafide (15) | TBD | TBD | TBD |

| Attack (15) | TBD | TBD | TBD |



\### Attack Breakdown



| Attack Type | Trusted | Suspicious | Fraud | n |

|-------------|---------|------------|-------|---|

| Video Replay | | | | 3 |

| Cloned Voice | | | | 3 |

| Virtual Camera | | | | 3 |

| Wrong Person | | | | 3 |

| Deepfake Video | | | | 3 |



\### Observed discrepancies between raw\_majority and ema bands



\*(list any session\_labels where `record\_session.py` printed a

band\_disagreement warning, and which band you used)\*



\### Known limitations of this run



\*(fill in after running — e.g. lighting conditions, which mic was used,

whether M1/M2's real models were loaded or fell back to mock scoring —

check console warnings from `IdentityMatcher`, `DeepfakeDetector`, and

`AASISTDetector` at startup for this)\*



\- On Windows, `SecurityMonitor` (main.py) opens a second `cv2.VideoCapture`

&#x20; on the same camera index used by the main loop, roughly every 8s (now

&#x20; 60s after mitigation). Windows DirectShow drivers do not reliably support

&#x20; two simultaneous capture handles on one device, causing intermittent

&#x20; frame corruption and an unreliable `injection\_risk\_score` (observed

&#x20; dropping to 0/100 even during legitimate sessions with no actual virtual

&#x20; camera present). This can artificially pull `trust\_score` down via the

&#x20; injection signal (weight 0.10) and may inflate BPCER. Mitigated by

&#x20; raising `cooldown\_sec` from 8.0 to 60.0 in `main.py`'s `SecurityMonitor`

&#x20; call, but not fully fixed — flagged as a pre-existing architectural

&#x20; limitation (see `SecurityMonitor`'s own docstring in `main.py`).



\---



\## Appendix A — Fusion-Logic Validation on Synthetic Signals (seed=42)



This section validates `security/fusion\_engine.py`'s arithmetic against

hand-constructed signal combinations. It does not involve real video,

audio, or camera hardware. It is evidence the weighted-sum, conflict

penalty, EMA smoothing, and identity cap are implemented correctly — not

evidence about detection accuracy against real attacks.



\### Test Protocol



\- \*\*Bonafide sessions (15, synthetic):\*\* identity similarity 0.85–0.98,

&#x20; visual deepfake prob 0.00–0.15, audio spoof prob 0.00–0.10, liveness

&#x20; passed, no virtual camera.

\- \*\*Attack sessions (15, synthetic; 3 per type):\*\*

&#x20; 1. Video Replay — liveness fails

&#x20; 2. Cloned Voice — synthetic voice, visual normal

&#x20; 3. Virtual Camera — OBS-style injection + deepfake stream

&#x20; 4. Wrong Person — different identity, similarity 0.20–0.50

&#x20; 5. Deepfake Video — direct deepfake feed, liveness fails



\### Results



| Metric | Value | Description |

|--------|-------|-------------|

| APCER | 0.0% | Attack sessions ending as "Trusted" (0/15) |

| BPCER | 0.0% | Bonafide sessions ending as "Potential Fraud" (0/15) |



\### Band Distribution



| Category | Trusted | Suspicious | Potential Fraud |

|----------|---------|------------|-----------------|

| Bonafide (15) | 15 | 0 | 0 |

| Attack (15) | 0 | 6 | 9 |



\### Attack Breakdown



| Attack Type | Avg Trust Score | Fraud | Suspicious | Trusted |

|-------------|-----------------|-------|------------|---------|

| Video Replay | 45.3 | 3/3 | 0/3 | 0/3 |

| Cloned Voice | 78.7 | 0/3 | 3/3 | 0/3 |

| Virtual Camera | 46.0 | 3/3 | 0/3 | 0/3 |

| Wrong Person | 78.0 | 0/3 | 3/3 | 0/3 |

| Deepfake Video | 43.0 | 3/3 | 0/3 | 0/3 |



\### Key Observations



\- Video Replay, Virtual Camera, and Deepfake Video are consistently

&#x20; blocked by conflict penalties.

\- Cloned Voice alone drops to Suspicious but not Potential Fraud —

&#x20; expected, since visual and injection signals remain clean and only

&#x20; 20% weight sits on audio.

\- Wrong Person is capped at Suspicious (79) by the identity-threshold-

&#x20; hardening rule (`identity\_trust < 0.60` → cap regardless of other

&#x20; signals), confirmed by `tests/test\_fusion\_parity.py::test\_wrong\_person\_attack\_blocked`.

\- BPCER = 0% on synthetic data — meaningless as a real-world guarantee

&#x20; until Section 1 confirms it against real lighting, real webcam noise,

&#x20; and real (imperfect) M1/M2 model outputs.



\### M3 Module Components (arithmetic only, not hardware-in-the-loop)



| Component | Status |

|-----------|--------|

| Fusion Engine (Weighted Sum) | Implemented, unit-tested |

| EMA Smoothing (alpha=0.3) | Implemented, unit-tested |

| Conflict Detection | Implemented, unit-tested |

| Conflict Penalty | Implemented, unit-tested |

| Identity Threshold Hardening | Implemented, unit-tested |

| Missing Signal Handling | Implemented, unit-tested |

| Explainability (Reason String) | Implemented |

| Validated against real M1/M2/M3 hardware | Pending Section 1 |



\### Known Limitations



1\. This appendix uses hand-picked signal values, not real detector

&#x20;  output — it cannot catch a case where M1's real classifier behaves

&#x20;  worse (or better) than these hypothetical numbers.

2\. Non-English audio samples have not been tested at all, real or

&#x20;  synthetic.

3\. Multi-speaker scenarios have not been tested.

4\. Low-light conditions are untested (no real camera involved here).



\---



\*Section 1 to be completed after a full 30-session live-stack run per

DEMO\_SCRIPT.md's scenarios. Appendix A stands as the fusion-logic

validation only and should not be re-cited as system evaluation once

Section 1 is filled in.\*

