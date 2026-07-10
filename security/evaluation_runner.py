# Evaluation Report — M3 Week 3

## APCER / BPCER System-Level Evaluation

&gt; **Note**: This evaluation follows the Week 3 Roadmap protocol (15 bonafide + 15 attack sessions).  
&gt; Results generated from mock signals with controlled randomness (seed=42).  
&gt; Real hardware evaluation with live M1/M2/M3 signals is scheduled for Day 7 full-team sync.

### Test Protocol

- **Bonafide Sessions**: 15
  - Simulated genuine users with natural signal variation.
  - Identity similarity: 0.85–0.98, Visual deepfake prob: 0.00–0.15, Audio spoof prob: 0.00–0.10.
  - Liveness challenge passed, no virtual camera detected.

- **Attack Sessions**: 15 (3 per attack type)
  1. **Video Replay** (3): Pre-recorded video played on phone; liveness fails.
  2. **Cloned Voice** (3): Synthetic voice; visual normal.
  3. **Virtual Camera** (3): OBS Virtual Camera with deepfake stream.
  4. **Wrong Person** (3): Different person attempts identity match.
  5. **Deepfake Video** (3): Direct deepfake video feed; liveness fails.

### Results

| Metric | Value | Description |
|--------|-------|-------------|
| APCER | 0.0% | Attack sessions ending as "Trusted" (0/15) |
| BPCER | 0.0% | Bonafide sessions ending as "Potential Fraud" (0/15) |

### Band Distribution

| Category | Trusted | Suspicious | Potential Fraud |
|----------|---------|------------|-----------------|
| Bonafide (15) | 15 | 0 | 0 |
| Attack (15) | 0 | 6 | 9 |

### Attack Breakdown

| Attack Type | Avg Trust Score | Blocked (Potential Fraud) | Suspicious | Trusted |
|-------------|-----------------|---------------------------|------------|---------|
| Video Replay | 45.3 | 3/3 | 0/3 | 0/3 |
| Cloned Voice | 78.7 | 0/3 | 3/3 | 0/3 |
| Virtual Camera | 46.0 | 3/3 | 0/3 | 0/3 |
| Wrong Person | 78.0 | 0/3 | 3/3 | 0/3 |
| Deepfake Video | 43.0 | 3/3 | 0/3 | 0/3 |

### Key Observations

- **Video Replay**, **Virtual Camera**, and **Deepfake Video** are consistently blocked by conflict penalties.
- **Cloned Voice** alone drops to "Suspicious" but not "Potential Fraud" — expected since visual and injection signals remain clean.
- **Wrong Person** is now capped at "Suspicious" by the identity threshold hardening rule (identity &lt; 0.6 → max Suspicious).
- **BPCER = 0%** confirms no legitimate users are unfairly blocked.

### M3 Module Components

| Component | Status |
|-----------|--------|
| Fusion Engine (Weighted Sum) | ✅ Working |
| EMA Smoothing (alpha=0.3) | ✅ Working |
| Conflict Detection | ✅ Working |
| Conflict Penalty (M3 enhancement) | ✅ Working |
| Identity Threshold Hardening | ✅ Working |
| Missing Signal Handling | ✅ Working |
| Explainability (Reason String) | ✅ Working |

### Known Limitations

1. Non-English audio samples have not been tested.
2. Multi-speaker scenarios have been tested only minimally.
3. Low-light conditions may reduce liveness challenge success rate.
4. Current evaluation uses mock signals — real M1/M2 hardware integration pending Day 7.
5. Cloned Voice and Wrong Person scenarios may land in "Suspicious" rather than "Potential Fraud" when acting in isolation.

### Dempster-Shafer Alternative (Stretch Goal)

- Planned as stretch goal per roadmap.
- Weighted sum + conflict penalty + identity threshold provides sufficient separation.
- D-S fusion can be added in a future iteration if formal uncertainty quantification is needed.
- For the pitch, weighted sum will be used (simpler to explain to judges).

---

*Prepared by: M3 (Security)*  
*Week 3, Day 7*