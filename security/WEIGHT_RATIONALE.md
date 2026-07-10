# Weight Rationale — DeepfakeGuard Trust Score Fusion (M3)

## Weight Distribution

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Identity Match | 0.25 | Identity verification — the most fundamental trust signal |
| Liveness Challenge | 0.25 | Liveness test — direct proof of human presence |
| Visual Deepfake | 0.20 | Visual deepfake detection — primary attack vector |
| Audio Spoof | 0.20 | Audio spoof detection — primary attack vector |
| Injection Risk | 0.10 | Virtual camera/injection detection — strong but narrow signal |

## Rationale

**Identity (0.25) + Liveness (0.25) = 0.50 (Half the score)**
- These two signals together provide the most direct proof of human presence.
- No user can be "Trusted" without passing the liveness challenge.
- The system cannot perform identity verification without an identity match.

**Visual (0.20) + Audio (0.20) = 0.40 (Next layer)**
- Primary adversarial signals: deepfake video and voice cloning.
- These two signals form the backbone of attack detection.
- Equal weight: each modality is an independent attack vector.

**Injection (0.10)**
- Virtual camera detection is a strong signal, however:
  - It overlaps significantly with visual deepfake detection.
  - It covers a narrower attack scenario (Arup-style attack).
- 0.10 weight strengthens this signal without dominating the others.

## Decision Rules

- **Trusted (≥80)**: All core signals strong, visual/audio risk low.
- **Suspicious (50–79)**: One or more signals weak but critical signals still standing.
- **Potential Fraud (&lt;50)**: Multiple signal failures or critical conflict detection.

## Conflict Cases

1. **identity_vs_visual_deepfake**: Identity matches but face is deepfake — sophisticated attack.
2. **liveness_vs_injection**: Liveness passed but virtual camera detected — possible bypass.

These conflicts are flagged as `conflict_detected: true` and trigger a special alert on the M4 dashboard.

## Conflict Penalty

When conflicts are detected, a penalty is applied to the weighted trust score:
- `identity_vs_visual_deepfake`: −10%
- `liveness_vs_injection`: −30% (more dangerous)

This ensures sophisticated attacks that fool one layer but not another are pushed into the "Potential Fraud" band.

---

*Prepared by: M3 (Security)*  
*Week 3, Day 7*