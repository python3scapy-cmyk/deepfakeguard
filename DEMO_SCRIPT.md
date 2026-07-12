# DeepfakeGuard — Demo Script (Week 3)

> **Status: UNREHEARSED.** This script has not yet been run end-to-end.
> It assumes M1's visual deepfake classifier and M2's audio module are
> live and POSTing to /score — as of this draft, neither is. Do not treat
> the "what to watch for" / "abort criteria" sections as verified behavior
> until at least one real run-through has happened.

This script defines exactly what happens on stage during the pitch demo. Every team member should read this before Week 4. No improvising.

---

## Pre-Demo Setup (5 minutes before)

1. Start the backend: `python backend/app.py`
2. Open the verifier dashboard: `http://localhost:5000` → `frontend/index.html`
3. Open the participant page on a second device: `http://<backend-ip>:5000/frontend/participant.html`
4. Verify WebRTC connection: both pages show "Streaming to verifier" / "Waiting for participant..."
5. Confirm all five panels are visible on the dashboard.
6. Reset the session via the dashboard's "Reset Session" button.

---

## Attack Scenario 1 — "Happy Path" (starts every demo)

**Duration:** 30-45 seconds

**What happens:**
1. Genuine team member sits in front of the camera on the participant device.
2. They accept the consent modal.
3. They complete enrollment (if not already enrolled) — [CONFIRM: exact
   enrollment flow/UI with M1/M3, this step is not yet verified].
4. They complete a blink liveness challenge.
5. All five panels stay green.
6. Trust score climbs to 85+. Band reads "Trusted."

**Narration:**
> "This is a legitimate user. All five signals clear. Trust score: 87. The identity panel shows a 92% match, the liveness challenge passed in under a second, and the visual, audio, and injection panels all show no anomalies."

**What to watch for:**
- Trust score should sit stably above 80 without flickering.
- Session log should show: `liveness_challenge` → `identity_check` → `band_change` to Trusted.
- If the score flickers into Suspicious, check the audio module — background noise can cause false positives.

**Abort criteria:** If trust score stays below 80 for more than 10 seconds on a genuine user, stop the demo and debug the audio module's threshold.

---

## Attack Scenario 2 — "Video Replay Attack"

**Duration:** 20-30 seconds

**What happens:**
1. Teammate holds up a phone/laptop playing a pre-recorded video of the enrolled person.
2. The liveness challenge is issued (blink or turn head).
3. The video cannot blink on command — liveness challenge fails.
4. Visual deepfake score rises (M1's EfficientNet detects synthetic artifacts).
5. Trust score drops to "Potential Fraud" (below 50).

**Narration:**
> "Same face, but pre-recorded video. The liveness challenge fails because a video can't blink on command. The visual deepfake detector also picks up compression artifacts from the screen replay. Both signals drop the trust score to Potential Fraud."

**What to watch for:**
- Liveness panel should show "Failed" with fail_reason "timeout" or "spoof_detected."
- Visual deepfake panel should rise toward red (score drops below 50).
- Reason string should mention both "liveness challenge failed" and "deepfake artifacts."
- Trust score should drop below 50 within 3-5 seconds of the challenge failing.

**Abort criteria:** If the visual deepfake panel doesn't react, M1's EfficientNet may not be running — check the cascade_stage_avg tooltip.

---

## Attack Scenario 3 — "Cloned Voice Attack"

**Duration:** 20-30 seconds

**What happens:**
1. The real enrolled person stays silent in front of the camera.
2. A cloned-voice clip of the enrolled person is played through a speaker in front of the mic.
3. Audio spoof score rises sharply (M2's AASIST+XGBoost ensemble detects synthetic voice).
4. Lip-sync score drops (mouth not moving, audio present).
5. Trust score drops. Reason string names "audio spoof risk."

**Narration:**
> "Voice cloned with a free TTS tool. The audio deepfake detector flags it in under two seconds. Notice the lip-sync score — the mouth isn't moving, but audio is present. That's a second independent signal."

**What to watch for:**
- Audio panel should rise sharply toward red (score drops below 50).
- Waveform visualiser should show erratic patterns (higher amplitude variance = more spoofed).
- Reason string should mention "synthetic voice confirmed."
- Trust score should drop below 50 within 2-3 seconds.

**Abort criteria:** If audio panel stays green, M2's AASIST model may not be connected — check the backend logs for `/score` POSTs from the audio module.

---

## Attack Scenario 4 — "Virtual Camera / Arup-Style Attack"

**Duration:** 15-25 seconds

**What happens:**
1. Activate OBS Virtual Camera with a deepfake video feed (or any pre-recorded video).
2. The participant page streams the OBS virtual camera feed instead of the real webcam.
3. Injection risk signal rises immediately (virtual camera driver detected by M3's device check).
4. Visual deepfake score also rises (the deepfake video itself has artifacts).
5. If the identity happens to match the enrolled person, a conflict alert fires.
6. Trust score drops. Dashboard explains both triggers.

**Narration:**
> "This is the Arup scenario. The entire video stream is fake — injected at the software level. Our injection detector catches the OBS Virtual Camera driver before any face analysis even runs. The visual deepfake detector confirms it. And if the identity happens to match, the conflict alert tells you this is a sophisticated attack — not just a random glitch."

**What to watch for:**
- Injection panel should show "Alert" with a red score (below 50).
- Visual deepfake panel should also rise.
- Conflict alert banner should appear: "Conflicting signals detected — identity matched but visual deepfake risk is elevated."
- Session log should show `injection_alert` event.
- Trust score should drop below 50 within 2 seconds of OBS activation.

**Abort criteria:** If injection panel doesn't react, M3's device check may not support the virtual camera name — check `device_check.py` known names list.

---

## Scenario Timing Cheat Sheet

**Target timings (not yet measured):**

| Scenario | Setup | Attack onset | Score drop | Expected final band |
|----------|-------|--------------|------------|---------------------|
| Happy Path | 10s | N/A | N/A | Trusted (≥80) |
| Video Replay | 5s | 5s | 3-5s | Fraud (<50) |
| Cloned Voice | 5s | 2s | 2-3s | Fraud (<50) |
| Virtual Camera | 5s | 2s | 2s | Fraud (<50) |

---

## Troubleshooting Quick Reference

**Problem:** Trust score stays at 0 / "Awaiting signals"
**Fix:** Check that all modules are POSTing to `http://localhost:5000/score`. Check backend console for received payloads.

**Problem:** Dashboard shows "Cannot reach backend"
**Fix:** Backend is not running. Run `python backend/app.py`.

**Problem:** Audio panel never updates from "Pending"
**Fix:** Audio module is not connected. Check that M2's script is running and POSTing to `/score`.

**Problem:** Conflict alert never appears
**Fix:** Requires both identity > 70 AND visual deepfake < 30 simultaneously. This only happens in the Virtual Camera scenario if the deepfake happens to match the enrolled identity. If not enrolled, run enrollment first.

**Problem:** Score drops too slowly (>5s)
**Fix:** EMA smoothing may be too aggressive. Check `EMA_ALPHA = 0.3` in `backend/app.py`. Lower = slower response.

**Problem:** Score flickers between bands on genuine user
**Fix:** Background noise or lighting changes. Check M2's audio quality threshold and M1's cascade_stage_avg. A stable genuine user should sit in Trusted without flickering.

---

## End-of-Demo Checklist

**Status: Not yet run.** Check items off only after a real rehearsal.

After every rehearsal, verify:
- [ ] All four scenarios were run at least once
- [ ] Session log contains entries for all expected event types
- [ ] No JavaScript errors in browser console
- [ ] Backend did not crash
- [ ] Trust score was stable in Happy Path (>80 for 30+ seconds)
- [ ] At least one scenario produced a conflict alert
- [ ] Reset button works and clears all state

---

*Version 1 — Week 3 Member 4 deliverable. Update after each rehearsal with observed timings and fixes.*
