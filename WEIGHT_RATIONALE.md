# DeepfakeGuard — Fusion Weight Rationale (Week 3)

This document explains why the trust-score fusion engine weights each signal the way it does. These weights will be scrutinised by judges. The reasoning must be defensible.

---

## Weight Distribution

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Identity match | 0.25 | Direct confirmation that the person in front of the camera is who they claim to be. This is the most fundamental biometric signal. |
| Liveness challenge | 0.25 | Direct confirmation that the subject is a live human responding to a real-time prompt, not a photo or video replay. |
| Visual deepfake | 0.20 | Detects synthetic face manipulation (GAN, diffusion, face-swap). This is a primary adversarial vector. |
| Audio deepfake | 0.20 | Detects synthetic voice (TTS, voice cloning, replay). This is a second primary adversarial vector. |
| Injection risk | 0.10 | Detects virtual cameras, frame injection, and stream tampering. Strong but narrower — overlaps with liveness (a replay attack also fails liveness). |

**Total:** 0.25 + 0.25 + 0.20 + 0.20 + 0.10 = **1.00**

---

## Why Identity and Liveness Get the Highest Weight (0.25 each)

Identity and liveness together constitute **half the score** because they are the most direct confirmations of a real, legitimate human presence:

- **Identity match** answers: "Is this the right person?" Without this, the system has no basis for trust. A wrong person should never be trusted, no matter how "real" their deepfake looks.
- **Liveness challenge** answers: "Is this a live human responding to a real-time prompt?" Without this, a pre-recorded video of the right person could pass. The challenge (blink, turn head) is a time-bound, unspoofable signal.

Together they form the **foundation layer** of the trust pyramid. The other three signals are **adversarial detection layers** that catch increasingly sophisticated attacks.

---

## Why Visual and Audio Deepfake Share the Next 40% (0.20 each)

Visual and audio deepfake detection are the **primary adversarial signals**:

- **Visual deepfake** catches face-swap, GAN-generated faces, and diffusion-based synthetic video. This is the most common deepfake attack vector in the news.
- **Audio deepfake** catches voice cloning (ElevenLabs, XTTS, etc.) and TTS-generated speech. Audio attacks are cheaper and easier to execute than video attacks.

They share equal weight because both are independent, high-impact attack vectors. A sophisticated attacker might compromise one but not the other. Giving them equal weight means the system degrades gracefully if either detector has a blind spot.

---

## Why Injection Risk Gets 10%

Injection risk (virtual camera detection, frame-timing anomaly) gets the **lowest weight** because:

1. **It is narrower.** It only catches software-level injection (OBS, ManyCam, etc.), not content-level deepfakes.
2. **It overlaps significantly with liveness.** A replayed stream injected via virtual camera will also fail the liveness challenge. Liveness already catches most of what injection catches, from a different angle.
3. **It is a strong corroborating signal, not a primary one.** When injection risk is high, it dramatically amplifies suspicion. But when it is low, it does not prove anything on its own.

However, 10% is **not zero**. In the Arup-style attack (full stream
injection), the injection signal is *designed* to fire before any face
analysis runs — this ordering has not yet been tested against a real
injection attempt, but it is the intended behavior the weight is meant
to support.

---

## Missing-Signal Handling

If a module has not reported in the last 3 seconds, it is marked as **missing** and assigned a **neutral placeholder of 0.5** (50 out of 100). This is critical:

- **0.5 is uncertainty, not distrust.** A missing module might be starting up, experiencing a temporary delay, or disconnected. Punishing it with 0.0 would unfairly crash the score.
- **The dashboard shows a spinner** on the missing panel, not a zero bar. This is transparent — the user knows the signal is absent, not fake.
- **Weights are still applied.** A missing signal still consumes its full weight with the neutral value, so the trust score is stable (not artificially inflated by dropping the weight).

Example: if audio is missing, the score is computed with audio_trust = 0.5, not skipped. The result is a slightly lower but still meaningful score.

---

## EMA Smoothing Rationale

The Exponential Moving Average (EMA) with **alpha = 0.3** smooths the trust score over time:

- **alpha = 0.3** means the new score contributes 30% and the previous EMA contributes 70%.
- This filters out one-frame noise (e.g., a motion blur causing a temporary deepfake spike) while still responding to genuine attacks within 2-3 update cycles.
- **Test:** If the score was 90 and a deepfake pushes the raw score to 30, the EMA crosses below 50 (Suspicious threshold) in approximately 2-3 cycles (~3-5 seconds at 1.5s polling).
- The EMA score is what the dashboard displays. The raw score is still available for debugging.

---

## Conflict Detection

Some signal combinations are **especially alarming** because they indicate a sophisticated attacker who fooled one layer but not another:

| Conflict | What it means | Why it matters |
|----------|-------------|----------------|
| identity > 70 AND visual < 30 | The face matches the enrolled person, but looks like a deepfake. | The attacker has a high-quality face-swap of the right person. The identity module is fooled; the visual detector is not. |
| liveness > 80 AND injection < 30 | The person passed the liveness challenge, but the stream is injected. | The liveness challenge was bypassed (e.g., real-time deepfake puppeteering). The injection detector caught the software-level tampering. |

These conflicts are surfaced as a **prominent alert banner** on the dashboard. They are not just score reductions — they are **explicit red flags** that demand human attention.

---

## Comparison: Weighted Sum vs. Dempster-Shafer

We also evaluated Dempster-Shafer (D-S) theory as an alternative fusion method. D-S handles conflicting evidence differently — instead of averaging, it amplifies conflict into explicit uncertainty.

**Decision:** We are keeping weighted sum for the pitch because it is simpler
to explain to judges and transparent to audit line-by-line.

**Status:** Dempster-Shafer is NOT implemented. It remains a stretch goal —
if pursued, it would need its own test pass against real attack data before
any comparison claim can be made.

---

## Validation Status

The weighted-sum arithmetic has been unit-tested against the fusion engine's
own logic (all-signals-high → Trusted; the roadmap's specified 60-point
Suspicious case) — that math is confirmed correct.

The four attack scenarios below have NOT yet been run against real M1/M2
output, because visual_deepfake_score and audio_deepfake_score have never
been POSTed by M1 or M2's actual modules. Once both are live, this section
should be replaced with real measured values from an actual session — not
hand-written estimates.

| Scenario | Status |
|----------|--------|
| Happy Path | Not yet run with real M1/M2 signals |
| Video Replay | Not yet run with real M1/M2 signals |
| Cloned Voice | Not yet run with real M1/M2 signals |
| Virtual Camera | Not yet run with real M1/M2 signals |

---

*Version 1 (draft) — Week 3. Weight rationale and fusion logic are implemented
and unit-tested. Scenario validation and D-S comparison are pending real
M1/M2 integration.*
