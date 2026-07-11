# Week 3 close-out — cari veziyyet (11 iyul)

## BAGLANDI
- P0-1 Audio mock -> real AASIST (297,866 params, backend=aasist_real)
  genuine 0.012 | macOS say TTS 1.000 | RTF 0.108
  model_backend payload-da; allow_mock=False strict rejim var
- Vision offline: models/deepfake-detector-v1 lokal, Wi-Fi sonulu halda siglip_hf
- main.py: mic buferi 2.0s -> 4.5s, AASIST-e 4s pencere

## YARIMCIQ
- clip_A_cloned.wav (ElevenLabs klonu) hele yaradilmayib
  Voice Lab -> Instant Voice Cloning -> MP3 endir -> afconvert
  Gozlenen spoof > 0.6; alinmasa pitch-de durust mehdudiyyet kimi qeyd et

## ACIQ BLOKER — P0-2 (en tehlukelisi)
security/fusion_engine.py iki davranisi tetbiq edir, backend/app.py hec birini:
  1. conflict penalty (-10% identity/visual, -30% liveness/injection)
  2. identity cap (identity < 0.6 -> bal 79-a mehdudlasir)
Netice: EVALUATION.md-in "wrong person bloklanir" iddiasi canli dashboard-da
islemir. Sehv adam liveness kecse ~78 alib Trusted-a surushe biler.
Fix: her iki bloku backend/app.py compute_trust()-a kocur + tests/test_fusion_parity.py

## ACIQ (kicik)
- P0-3 real APCER/BPCER (15+15 sessiya)
- P0-4 per-modul eval (M1, M2 ucun 10+10)
- P1-6 frontend: test-payload duymesi demo-da aciqdir; waveform dekorativdir
