# DeepfakeGuard — Consent Modal Copy (v2)

Architecture facts confirmed by M1/M2/M3 (as of this draft):
- No photos leave the device — face frames are processed into an embedding locally.
- No recordings are stored — audio is processed into a voice embedding in memory only.
- No external API calls — all models (M1/M2/M3) run entirely on-device.
- **The one-time enrollment face embedding IS written to disk**
  (`security/enrolled_face.json`), so verification sessions can reuse it
  without re-enrolling each time. This is the sole exception to "nothing is
  persisted" and must be disclosed, not glossed over.

All text below depends on these facts remaining true. If any of them change
(e.g. a model moves to a cloud endpoint, or the enrollment file starts being
synced anywhere off-device), this file and the Privacy Notice must be revised
before the next release.

---

## 1. WHAT WE COLLECT

- A one-time enrollment face embedding, stored on disk so you don't have to
  re-enroll every session (not a photo)
- A per-session live face embedding, held in memory only (not a photo)
- A voice embedding, held in memory only (not a recording)
- Video stream metadata for liveness verification

## 2. WHY

To verify your identity and detect deepfake impersonation in real time.

## 3. HOW LONG WE KEEP IT

- **Enrollment embedding:** kept on disk until you delete it or re-enroll.
  It is intentionally persistent — that's what lets you skip re-enrolling
  every session.
- **Everything else** (live embeddings, voice embeddings, session metadata):
  session only. Discarded when the session ends. Never written to disk.

## 4. YOUR RIGHTS

- You may withdraw consent at any time by ending the session — this stops
  all in-session data collection immediately.
- You may request deletion of your enrollment embedding at any time; this is
  a manual action (delete the file, or ask whoever operates this instance
  to do so), not an automatic one.
- You may request deletion of any in-session data, though this data is
  already deleted automatically at session end.

## 5. BIOMETRIC DATA NOTICE (BIPA-required block)

> BIOMETRIC DATA NOTICE: We are collecting biometric identifiers (a
> persisted enrollment face embedding, plus in-session face and voice
> embeddings) to verify your identity and detect deepfake impersonation. The
> enrollment embedding is retained on disk until you request its deletion or
> re-enroll. All other session data is retained only for the duration of the
> session and is deleted when the session ends. You may withdraw consent to
> an active session at any time by ending the session, and may request
> deletion of your stored enrollment embedding at any time.

**Checkpoint:** This version discloses the persisted enrollment embedding
explicitly, rather than implying everything is session-only. BIPA requires
written notice of what is collected, why, and for how long — a notice that
describes session-only retention while a biometric identifier is actually
persisted on disk would not satisfy that requirement honestly.

## 6. DECLINE STATE COPY

> Identity verification required to proceed.
>
> This service requires camera and microphone access to verify your
> identity and detect deepfake attempts in real time. Without consent, you
> cannot continue.

---

*Version 2 — revised [fill in date] to correct a discrepancy where this
document and the Privacy Notice both stated no data was written to disk,
while `security/enrolled_face.json` (a persisted enrollment embedding) was
present in the repository. Any further edit to this file should be a
reviewed commit, not a casual copy change — treat it as a legal artefact.*
