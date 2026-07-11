# M2 — Audio anti-spoofing: per-module evaluation

**Date:** 2026-07-11
**Model:** AASIST (clovaai/aasist), ASVspoof2019 pretrained checkpoint, 297,866 params
**Backend flag:** `model_backend = "aasist_real"` (verified non-mock in payload)
**Threshold:** 0.5 (same as live pipeline)
**Hardware:** MacBook Air (Apple Silicon), CPU inference

## Results

| Metric | Value |
|---|---|
| N bonafide | 10 |
| N spoof | 8 |
| **APCER** | **0.0%** (0/8 attacks passed as genuine) |
| **BPCER** | **0.0%** (0/10 genuine rejected) |
| bonafide mean / max | 0.019 / 0.038 |
| spoof mean / min | 0.995 / 0.959 |
| Separation gap | 0.92 |
| RTF (50 runs) | mean 0.102, p95 0.135, max 0.162 |

## Protocol

**Bonafide (n=10):** independently recorded 5s utterances, 16kHz mono, varied
conditions — close mic, normal distance, 1-2m distance, background noise,
near-whisper, loud/emotional speech.

**Spoof (n=8):** macOS `say` TTS, 8 distinct voices (Alex, Daniel, Fred, Karen,
Moira, Rishi, Samantha, Tessa), identical attack phrase, converted to 16kHz mono.

## Limitations — state these in the pitch

1. **Attack samples are macOS `say` TTS only.** This is concatenative/older-generation
   synthesis. Modern neural voice cloning (ElevenLabs, XTTS) was NOT tested — AASIST
   is trained on ASVspoof2019 and may not generalize to 2026-era cloning.
2. **Single speaker, single session** for bonafide samples (one person, one room,
   one microphone). BPCER on a diverse population is unknown.
3. **No ASVspoof eval-set numbers** — this is an in-the-wild smoke test, not a
   benchmark result.
