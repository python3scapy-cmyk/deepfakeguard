\# Identity Module — FaceNet/ArcFace 1:1 Identity Matching



\## What it does

Enrolls a reference face at session start, then continuously verifies

that the live face matches the enrolled identity using ArcFace embeddings

and cosine similarity.



\## How to run



\### Step 1 - Enroll your face:

python enroll.py



\### Step 2 - Run live verification:

python verify.py



\## Output payload

{

&#x20; "identity\_match": true,

&#x20; "similarity\_score": 0.88,

&#x20; "threshold\_used": 0.6,

&#x20; "face\_detected": true,

&#x20; "multiple\_faces\_detected": false,

&#x20; "confidence": "high",

&#x20; "latency\_ms": 1320.9,

&#x20; "timestamp": 1719000100.22

}



\## Edge cases handled

\- No face detected: identity\_match=False, face\_detected=False

\- Multiple faces: multiple\_faces\_detected=True, identity\_match=False

\- Uncertain score (0.50-0.60): confidence="low"

\- Photo attack: ArcFace did not recognize phone screen as a face



\## Performance

\- Verification runs every 15th frame at 320x320 resolution

\- Latency: 1300-2800ms on integrated CPU (hardware dependent)

\- Impostor substitution caught within \~3 seconds at 15fps



\## Known limitations

\- ArcFace without liveness may match a high-quality printed photo

&#x20; held very close to camera. M1's liveness challenge is the defence

&#x20; layer for this attack vector.

\- Latency is high on CPU-only machines; GPU would reduce to under 100ms



\## Notes

\- Stores a 512-dimensional mathematical embedding, NOT a photo

\- GDPR Article 9 data-minimisation compliant

\- Embeddings are session-only, not persisted to disk after session ends

