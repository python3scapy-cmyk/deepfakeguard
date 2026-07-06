# DeepfakeGuard — Consent Modal Copy (v1)

Architecture facts confirmed by M1/M2/M3 (as of this draft):
- No photos leave the device — face frames are processed into an embedding locally.
- No recordings are stored — audio is processed into a voice embedding in memory only.
- No external API calls — all models (M1/M2/M3) run entirely on-device.

All text below depends on these facts remaining true. If any of them change
(e.g. a model moves to a cloud endpoint), this file and the Privacy Notice
must be revised before the next release.

---

## 1. WHAT WE COLLECT

- A mathematical face embedding (not a photo)
- A voice embedding (not a recording)
- Video stream metadata for liveness verification

## 2. WHY

To verify your identity and detect deepfake impersonation in real time.

## 3. HOW LONG WE KEEP IT

Session only. Embeddings are discarded when the session ends. Nothing is
written to disk or sent to a server.

## 4. YOUR RIGHTS

- You may withdraw consent at any time by ending the session.
- You may request deletion of any session data.

## 5. BIOMETRIC DATA NOTICE (BIPA-required block)

> BIOMETRIC DATA NOTICE: We are collecting biometric identifiers (face and
> voice embeddings) to verify your identity and detect deepfake
> impersonation. This data will be retained only for the duration of this
> session and will be permanently deleted when the session ends. You may
> withdraw consent at any time by ending the session.

**Checkpoint:** Including this verbatim satisfies BIPA's written notice,
specific purpose, and retention period requirements in one block.

## 6. DECLINE STATE COPY

> Identity verification required to proceed.
>
> This service requires camera and microphone access to verify your
> identity and detect deepfake attempts in real time. Without consent, you
> cannot continue.

---

*Version 1 — drafted [fill in date]. Treat as a legal artefact: any edit
should be a reviewed commit, not a casual copy change.*
