#!/usr/bin/env python3
"""
Spoken-phrase liveness challenge ("Say: magic apples") - the audio
counterpart to the vision challenges (blink / turn head).

Verification backends, tried in order:
  1. faster-whisper  (pip install faster-whisper) - CTranslate2, CPU-fast,
     'tiny' model ~75MB int8, transcribes a 4s clip in well under a second.
  2. openai-whisper  (pip install openai-whisper) - needs torch (already a
     project dependency for SigLIP), 'tiny' checkpoint.
  3. Energy-only fallback - can only confirm that SPEECH HAPPENED in the
     window, not WHAT was said. Results are clearly labeled
     mode='energy_only' so the UI/demo never over-claims.

Phrase matching is deliberately forgiving: lowercase, punctuation
stripped, then per-word fuzzy matching (difflib) - ASR writing
"magik apples" or "magic apple" must still pass a human who clearly
tried. An attacker replaying a recorded clip fails not here but at the
AASIST spoof scorer, which runs on the SAME clip (see engine.py).
"""
import difflib
import random
import re
import string
import threading
import time

import numpy as np

# Short, phonetically distinct, easy for non-native speakers.
PHRASE_POOL = [
    "magic apples",
    "hello world",
    "open sesame",
    "blue horizon",
    "silver mountain",
    "seven green stars",
    "golden river",
    "quiet thunder",
]

WORD_MATCH_RATIO = 0.72   # difflib ratio for a single word to count as matched
PHRASE_PASS_RATIO = 0.6   # fraction of phrase words that must be matched


def _normalize(text):
    text = text.lower().strip()
    return re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", text)


class SpeechChallengeVerifier:
    """Lazy-loads an ASR backend on first verify() call (not at import or
    construction) so backend startup stays fast."""

    def __init__(self, model_size="tiny"):
        self.model_size = model_size
        self._backend = None          # "faster_whisper" | "whisper" | "energy_only"
        self._model = None
        self._load_lock = threading.Lock()

    # ---------------- challenge lifecycle ----------------
    def new_phrase(self, exclude=None):
        pool = [p for p in PHRASE_POOL if p != exclude] or PHRASE_POOL
        return random.choice(pool)

    # ---------------- ASR loading ----------------
    def _ensure_loaded(self):
        with self._load_lock:
            if self._backend is not None:
                return
            try:
                from faster_whisper import WhisperModel
                print(f"[SPEECH] Loading faster-whisper '{self.model_size}' "
                      f"(first run downloads ~75MB)...")
                self._model = WhisperModel(self.model_size, device="cpu",
                                           compute_type="int8")
                self._backend = "faster_whisper"
                print("[SPEECH] faster-whisper ready")
                return
            except Exception as e:
                print(f"[SPEECH][WARN] faster-whisper unavailable ({e})")
            try:
                import whisper
                print(f"[SPEECH] Loading openai-whisper '{self.model_size}'...")
                self._model = whisper.load_model(self.model_size)
                self._backend = "whisper"
                print("[SPEECH] openai-whisper ready")
                return
            except Exception as e:
                print(f"[SPEECH][WARN] openai-whisper unavailable ({e})")
            print("[SPEECH][WARN] No ASR backend - falling back to "
                  "energy-only speech detection (phrase content NOT verified). "
                  "Fix: pip install faster-whisper")
            self._backend = "energy_only"

    # ---------------- transcription ----------------
    def _transcribe(self, pcm_16k_f32):
        if self._backend == "faster_whisper":
            segments, _info = self._model.transcribe(
                pcm_16k_f32, language="en", beam_size=1,
                vad_filter=True, condition_on_previous_text=False)
            return " ".join(seg.text for seg in segments).strip()
        if self._backend == "whisper":
            result = self._model.transcribe(
                pcm_16k_f32.astype(np.float32), language="en", fp16=False)
            return (result.get("text") or "").strip()
        return None  # energy_only

    # ---------------- verification ----------------
    def verify(self, pcm_16k_f32, expected_phrase):
        """Returns a dict:
        {passed, mode, transcript, expected_phrase, match_ratio,
         speech_energy, latency_ms}"""
        start = time.time()
        self._ensure_loaded()

        pcm = np.asarray(pcm_16k_f32, dtype=np.float32).flatten()
        # Simple energy gate first: reject silence outright regardless of
        # backend (whisper hallucinated text on silence is a known trap).
        rms = float(np.sqrt(np.mean(pcm ** 2))) if pcm.size else 0.0
        frame_len = 1600  # 100ms @ 16k
        n_frames = max(1, len(pcm) // frame_len)
        frames = pcm[:n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        voiced_ratio = float(np.mean(frame_rms > max(0.008, rms * 0.5)))
        has_speech = rms > 0.005 and voiced_ratio > 0.15

        base = {
            "mode": self._backend,
            "expected_phrase": expected_phrase,
            "speech_energy": round(rms, 4),
            "transcript": None,
            "match_ratio": 0.0,
        }

        if not has_speech:
            base.update({"passed": False,
                         "reason": "no speech detected in the clip"})
            base["latency_ms"] = round((time.time() - start) * 1000, 1)
            return base

        if self._backend == "energy_only":
            # Can't verify content - pass on clear speech, but say so.
            base.update({"passed": True,
                         "reason": "speech detected (phrase content NOT "
                                   "verified - no ASR backend installed)"})
            base["latency_ms"] = round((time.time() - start) * 1000, 1)
            return base

        try:
            transcript = self._transcribe(pcm) or ""
        except Exception as e:
            print(f"[SPEECH][ERROR] transcription failed: {e}")
            base.update({"passed": False,
                         "reason": f"transcription error: {e}"})
            base["latency_ms"] = round((time.time() - start) * 1000, 1)
            return base

        expected_words = _normalize(expected_phrase).split()
        heard_words = _normalize(transcript).split()
        matched = 0
        for ew in expected_words:
            best = max((difflib.SequenceMatcher(None, ew, hw).ratio()
                        for hw in heard_words), default=0.0)
            if best >= WORD_MATCH_RATIO:
                matched += 1
        ratio = matched / len(expected_words) if expected_words else 0.0
        passed = ratio >= PHRASE_PASS_RATIO

        base.update({
            "passed": passed,
            "transcript": transcript,
            "match_ratio": round(ratio, 2),
            "reason": (f"heard '{transcript}' - matched {matched}/"
                       f"{len(expected_words)} words"),
        })
        base["latency_ms"] = round((time.time() - start) * 1000, 1)
        return base
