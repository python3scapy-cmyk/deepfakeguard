#!/usr/bin/env python3
"""
Faza 0 smoke test: prove the headless AnalysisEngine works with NO camera,
NO backend, NO GUI window - exactly the way backend/app.py will drive it
in Faza 1.

Run:  python engine_selftest.py

Pass criteria:
  1. Engine constructs (shared models load or cleanly fall back to mock).
  2. 60 synthetic frames are processed without an exception.
  3. At least one fusion payload is emitted (EMIT_EVERY=10 -> expect ~6).
  4. Payload contains every field backend's receive_fusion() reads.
  5. Session manager caps concurrent sessions at MAX_ACTIVE_SESSIONS.

NOTE: first ever run may download the SigLIP deepfake model (~370MB,
cached afterwards under ~/.cache/huggingface). If the download or a
dependency is unavailable the detectors fall back to their labeled mock
modes and this test still passes - it validates PLUMBING, not accuracy.
"""
import sys
import numpy as np
import cv2

from engine import AnalysisEngine, get_engine, drop_session, ENGINES, MAX_ACTIVE_SESSIONS

REQUIRED_TOP_LEVEL = [
    "module", "session_id", "timestamp", "face_detected", "multi_face_alert",
    "trust_score", "trust_band", "lowest_signal", "signals",
    "hard_deny_reasons", "verdict", "challenge_result",
]
REQUIRED_SIGNALS = [
    "liveness_challenge_passed", "visual_deepfake_probability",
    "audio_spoof_probability", "lip_sync_score", "anti_spoof_2d_raw",
    "anti_spoof_2d_sustained", "identity_similarity",
    "injection_risk_score", "virtual_camera_detected",
]


def make_synthetic_frame(i):
    """Same cartoon face main.py's create_test_video draws (no file I/O)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 100, (200, 180, 160), -1)
    cv2.circle(frame, (290, 220), 15, (50, 50, 50), -1)
    cv2.circle(frame, (350, 220), 15, (50, 50, 50), -1)
    cv2.ellipse(frame, (320, 270), (40, 20), 0, 0, 180, (100, 50, 50), 3)
    noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.putText(frame, f"Frame: {i}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main():
    failures = []

    # ---- 1+2: construct and process 60 frames ----
    print("=" * 60)
    print("  ENGINE SELFTEST - Faza 0")
    print("=" * 60)
    eng = AnalysisEngine(session_id="selftest_session", remote=True)

    emissions = []
    for i in range(60):
        frame = make_synthetic_frame(i)
        try:
            fusion = eng.process_frame(frame)
        except Exception as e:
            failures.append(f"process_frame raised on frame {i}: {e}")
            import traceback; traceback.print_exc()
            break
        if fusion is not None:
            emissions.append(fusion)
            print(f"  frame {i:>3}: EMIT  trust={fusion['trust_score']:>5} "
                  f"band={fusion['trust_band']:<10} verdict={fusion['verdict']}")

    # ---- 3: emission count ----
    if not emissions:
        failures.append("no fusion payload was emitted in 60 frames")
    else:
        print(f"\n[OK] {len(emissions)} fusion payload(s) emitted "
              f"(expected ~{60 // 10})")

    # ---- 4: field parity with backend's receive_fusion ----
    if emissions:
        last = emissions[-1]
        for key in REQUIRED_TOP_LEVEL:
            if key not in last:
                failures.append(f"missing top-level field: {key}")
        for key in REQUIRED_SIGNALS:
            if key not in last.get("signals", {}):
                failures.append(f"missing signals field: {key}")
        if not failures:
            print("[OK] payload has every field backend/app.py reads")
        if last["signals"]["injection_risk_score"] is not None:
            failures.append("remote session must report injection_risk_score=None")

    eng.close()

    # ---- 5: session manager cap ----
    made = [get_engine(f"cap_test_{k}") for k in range(MAX_ACTIVE_SESSIONS + 2)]
    active = sum(1 for m in made if m is not None)
    if active != MAX_ACTIVE_SESSIONS:
        failures.append(f"session cap broken: {active} active, "
                        f"expected {MAX_ACTIVE_SESSIONS}")
    else:
        print(f"[OK] session cap enforced at {MAX_ACTIVE_SESSIONS}")
    for k in range(MAX_ACTIVE_SESSIONS + 2):
        drop_session(f"cap_test_{k}")
    if ENGINES:
        failures.append(f"sessions not cleaned up: {list(ENGINES)}")

    # ---- verdict ----
    print("\n" + "=" * 60)
    if failures:
        print("  SELFTEST: FAIL")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)
    print("  SELFTEST: PASS - engine is ready for Faza 1 (backend wiring)")
    print("=" * 60)


if __name__ == "__main__":
    main()
