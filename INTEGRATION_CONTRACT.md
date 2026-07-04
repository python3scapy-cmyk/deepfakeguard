\# DeepfakeGuard — Integration Contract



Each module POSTs its result to `http://localhost:5000/score` as JSON.

Every payload must include a `module` field so the backend knows which

signal it is. All \*\_score fields are 0–100 where higher = more trustworthy.



\---



\## Member 1 — Vision (identity + liveness + visual deepfake)



POST /score

```json

{

&#x20; "module": "vision",

&#x20; "session\_id": "live\_webrtc\_call\_xyz\_123",

&#x20; "timestamp": "2026-06-29T22:15:00Z",

&#x20; "face\_detected": true,

&#x20; "multi\_face\_alert": false,

&#x20; "identity\_score": 89,

&#x20; "liveness\_score": 89,

&#x20; "visual\_deepfake\_score": 85,

&#x20; "verdict": "REAL"

}

```



\*\*Score conversion rules:\*\*

\- `identity\_score` — 0–100 directly

\- `liveness\_score` — `texture\_liveness\_score × 100`

\- `visual\_deepfake\_score` — derive from your existing artifact detection

\- Drop the `fusion\_engine` block — that is the backend's job, not this module's



\---



\## Member 2 — Audio (audio deepfake)



POST /score

```json

{

&#x20; "module": "audio",

&#x20; "session\_id": "live\_webrtc\_call\_xyz\_123",

&#x20; "timestamp": "2026-06-29T22:15:00Z",

&#x20; "voice\_detected": true,

&#x20; "audio\_deepfake\_score": 88,

&#x20; "verdict": "REAL"

}

```



\*\*Score conversion rules:\*\*

\- `audio\_deepfake\_score` = `(1 - xgboost\_synthetic\_prob) × 100`

\- Example: synthetic\_prob 0.12 → audio\_deepfake\_score 88

\- Drop the `fusion\_engine` block — that is the backend's job, not this module's



\---



\## Member 3 — Security (injection / stream integrity)



POST /score

```json

{

&#x20; "module": "security",

&#x20; "session\_id": "live\_webrtc\_call\_xyz\_123",

&#x20; "timestamp": "2026-06-29T22:15:00Z",

&#x20; "virtual\_camera\_detected": false,

&#x20; "device\_name": "FaceTime HD Camera",

&#x20; "injection\_risk\_score": 88,

&#x20; "frame\_timing\_anomaly\_score": 4,

&#x20; "verdict": "REAL"

}

```



\*\*Score conversion rules:\*\*

\- `injection\_risk\_score` = `round((1 - anomaly\_score) \* 100)`

&#x20; - anomaly\_score 0.04 → injection\_risk\_score 96

&#x20; - anomaly\_score 0.80 → injection\_risk\_score 20

\- `frame\_timing\_anomaly\_score` = `round(coefficient\_of\_variation \* 100)`

\- `virtual\_camera\_detected` — pass through directly as true or false

\- `verdict` = `"FAKE"` if virtual\_camera\_detected or injection\_risk\_score < 50, else `"REAL"`



\---



\## Backend rules (Member 4)



\- Receives all three POSTs separately via `/score`

\- Stores latest payload per module in `latest\_scores` dict

\- Computes trust band: score >= 80 → Trusted, >= 50 → Suspicious, else Fraud

\- Fusion logic lives here only — no module should compute a final decision itself

\- GET `/scores` returns all latest signals to the dashboard



\---



\## Common rules for all modules



\- POST to `http://localhost:5000/score` 

\- Always include `module`, `session\_id`, `timestamp`, `verdict`

\- All score fields are integers 0–100, higher = more trustworthy

\- Send updates continuously while a session is active (suggested: every 1–2 seconds)

\- Do NOT include a `fusion\_engine` block in your payload

