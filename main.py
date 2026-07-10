#!/usr/bin/env python3
"""
DeepfakeGuard - Week 3 Fusion Engine

Single entry point that wires together:
  - vision.face_detector      (face detection + 2D anti-spoof heuristics)
  - vision.liveness_engine    (blink / head-turn challenge state machine)
  - vision.deepfake_detector  (cascaded visual deepfake classifier)
  - audio_module.lip_sync     (mouth/audio correlation proxy)
  - audio_module.aasist_detector (AASIST+XGBoost audio spoof ensemble)

and combines all five signals into one live trust score.

Run with:  python3 main.py
"""
import json
import os
import platform
import time
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np
import argparse
import requests

from vision.face_detector import FaceLandmarkDetector
from vision.liveness_engine import LivenessChallengeEngine
from vision.deepfake_detector import DeepfakeDetector
from audio_module.lip_sync import LipSyncDetector
from audio_module.aasist_detector import AASISTDetector


def _has_faces(faces):
    if faces is None:
        return False
    if isinstance(faces, np.ndarray):
        return faces.size > 0
    return len(faces) > 0


class DeepfakeGuardSystem:
    WEIGHTS = {
        "identity": 0.20,
        "liveness": 0.25,
        "visual_deepfake": 0.25,
        "audio_spoof": 0.20,
        "lip_sync": 0.10,
    }
    BACKEND_URL = "http://localhost:5000"

    def post_to_backend(self, fusion_json_str):
        """POST the fusion JSON to the Flask backend's /fusion endpoint.
        Failures are logged but never crash the camera loop."""
        if self.no_backend:
            return
        try:
            resp = requests.post(
                self.BACKEND_URL + "/fusion",
                data=fusion_json_str,
                headers={"Content-Type": "application/json"},
                timeout=1.0
            )
            if resp.status_code != 200:
                print(f"[WARN] Backend returned {resp.status_code}: {resp.text[:120]}")
        except requests.exceptions.ConnectionError:
            pass  # backend not running - silent
        except Exception as e:
            print(f"[WARN] Backend post failed: {e}")

    def __init__(self):
        print("=" * 60)
        print("  DEEPFAKEGUARD - WEEK 3 FUSION SYSTEM")
        print("  Vision (anti-spoof + liveness + deepfake) + Audio + LipSync")
        print("=" * 60)

        self.vision_detector = FaceLandmarkDetector()
        self.liveness_engine = LivenessChallengeEngine()
        self.deepfake_detector = DeepfakeDetector()  # pass model_path="models/xxx.pth" once you have real weights
        self.lip_sync = LipSyncDetector(window_ms=200)
        self.audio_detector = AASISTDetector()

        self.session_id = f"session_{int(time.time())}"
        self.challenge_active = False
        self.challenge_cooldown = 0
        self.frame_count = 0
        self.use_audio = False
        self.no_backend = False
        # Rolling buffer of recent per-frame spoof flags. A single noisy
        # frame (bad lighting moment, motion blur) shouldn't be able to
        # trigger a hard DENIED by itself -- we require a clear majority
        # over a short recent window before treating it as a real spoof.
        self.spoof_history = deque(maxlen=15)

    # ---------------- fusion ----------------
    def compute_trust_score(self, scores):
        weighted_sum, total_weight = 0.0, 0.0
        for key, weight in self.WEIGHTS.items():
            val = scores.get(key)
            if val is not None:
                weighted_sum += val * weight
                total_weight += weight

        if total_weight == 0:
            return {"trust_score": 50.0, "band": "SUSPICIOUS",
                    "lowest_signal": "none", "lowest_score": 0}

        trust_score = weighted_sum / total_weight
        if trust_score >= 80:
            band = "TRUSTED"
        elif trust_score >= 60:
            band = "CAUTION"
        elif trust_score >= 40:
            band = "SUSPICIOUS"
        else:
            band = "DENIED"

        valid = [(k, v) for k, v in scores.items() if v is not None]
        lowest = min(valid, key=lambda kv: kv[1]) if valid else ("none", 0)

        return {
            "trust_score": round(trust_score, 1),
            "band": band,
            "lowest_signal": lowest[0],
            "lowest_score": lowest[1],
        }

    def generate_json(self, liveness_payload, vision_payload, deepfake_result,
                       audio_result, sync_result, trust_data, smoothed_spoof_flag):
        challenge = liveness_payload.get("challenge_result")
        challenge_passed = bool(challenge and challenge.get("challenge_passed"))

        spoof_analysis = vision_payload.get("spoof_analysis") or {}
        # Use the temporally-smoothed (majority-vote-over-recent-frames)
        # spoof flag for the actual verdict -- not the single-frame raw
        # heuristic, which is noisy. The raw per-frame value is still
        # available in spoof_analysis for debugging if needed.
        anti_spoof_2d_flag = smoothed_spoof_flag
        face_detected = liveness_payload.get("face_detected", False)
        multi_face = liveness_payload.get("multi_face", False)

        # Hard overrides -- these are NOT averaged into the trust score,
        # they force a denial outright. A weighted-average trust score can
        # still land in "GRANTED" territory even when one strong signal
        # (e.g. "this is a phone screen") screams spoof; that's not
        # acceptable for an access-control verdict.
        hard_deny_reasons = []
        if not face_detected:
            hard_deny_reasons.append("no_face_detected")
        if multi_face:
            hard_deny_reasons.append("multiple_faces")
        if anti_spoof_2d_flag:
            hard_deny_reasons.append("2d_spoof_detected_sustained")

        if hard_deny_reasons:
            verdict = "ACCESS DENIED"
        else:
            verdict = "ACCESS GRANTED" if trust_data["band"] in ("TRUSTED", "CAUTION") else "ACCESS DENIED"

        data = {
            "module": "fusion",
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "face_detected": face_detected,
            "multi_face_alert": multi_face,
            "trust_score": trust_data["trust_score"],
            "trust_band": trust_data["band"],
            "lowest_signal": trust_data["lowest_signal"],
            "signals": {
                "liveness_challenge_passed": challenge_passed,
                "visual_deepfake_probability": (
                    deepfake_result["deepfake_probability"] if deepfake_result else None
                ),
                "audio_spoof_probability": (
                    audio_result["spoof_probability"] if audio_result else None
                ),
                "lip_sync_score": sync_result.get("lip_sync_score"),
                "anti_spoof_2d_raw": spoof_analysis.get("is_spoof", False),
                "anti_spoof_2d_sustained": anti_spoof_2d_flag,
            },
            "hard_deny_reasons": hard_deny_reasons,
            "verdict": verdict,
            "challenge_result": challenge,
        }
        return json.dumps(data, indent=2)

    # ---------------- test helper ----------------
    def create_test_video(self, path="test_video.mp4", frames=150):
        print(f"[INFO] Creating synthetic test video: {path}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, 30.0, (640, 480))
        for i in range(frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(frame, (320, 240), 100, (200, 180, 160), -1)
            cv2.circle(frame, (290, 220), 15, (50, 50, 50), -1)
            cv2.circle(frame, (350, 220), 15, (50, 50, 50), -1)
            cv2.ellipse(frame, (320, 270), (40, 20), 0, 0, 180, (100, 50, 50), 3)
            noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(frame)
        out.release()
        print(f"[OK] {path} ready")
        return path

    def _candidate_backends(self):
        """Return a list of (backend_id, label) to try, in order, based on OS.
        Falls back to cv2.CAP_ANY (let OpenCV pick) last on every platform,
        so this degrades gracefully even on an OS we didn't special-case."""
        system = platform.system()
        backends = []
        if system == "Darwin":
            backends.append((cv2.CAP_AVFOUNDATION, "AVFoundation"))
        elif system == "Windows":
            backends.append((cv2.CAP_DSHOW, "DirectShow"))
            backends.append((cv2.CAP_MSMF, "Media Foundation"))
        elif system == "Linux":
            backends.append((cv2.CAP_V4L2, "V4L2"))
        backends.append((cv2.CAP_ANY, "auto"))
        return backends

    def _try_read_with_retry(self, cap, attempts=10, delay=0.1):
        """Some backends need a couple of frames to warm up before read()
        succeeds -- retry briefly instead of giving up on the first miss."""
        for _ in range(attempts):
            ret, frame = cap.read()
            if ret and frame is not None:
                return frame
            time.sleep(delay)
        return None

    def _open_camera(self):
        print("\nOpening camera...")
        print("(On Mac: System Settings -> Privacy & Security -> Camera -> enable Terminal)")
        print("(On Windows: Settings -> Privacy & security -> Camera -> allow desktop apps)")
        print("(On Linux: make sure your user is in the 'video' group and no other app has /dev/video0 open)")

        indices_to_try = [0, 1, 2]
        for backend_id, backend_label in self._candidate_backends():
            for idx in indices_to_try:
                cap = cv2.VideoCapture(idx, backend_id)
                if not cap.isOpened():
                    cap.release()
                    continue
                test_frame = self._try_read_with_retry(cap)
                if test_frame is None:
                    cap.release()
                    continue
                print(f"Camera OK | index {idx} | backend: {backend_label} | frame shape: {test_frame.shape}")
                # NOTE: we deliberately do NOT call cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT)
                # here. Forcing a resolution change right after opening makes some
                # backends/cameras (esp. on macOS AVFoundation, but also some
                # Windows webcams) fail the *next* read(), which the main loop
                # would otherwise misread as "end of stream". We just use
                # whatever native resolution the camera gives us.
                return cap

        print("\n" + "=" * 50)
        print("  CAMERA COULD NOT BE OPENED")
        print("=" * 50)
        print("Likely causes:")
        print("  - OS-level camera permission not granted to this terminal/app")
        print("  - Another app is using the camera (Zoom/Teams/Meet/Chrome/etc.)")
        print("  - No camera at index 0/1/2, or a driver issue")
        print("Try: closing other apps using the camera, checking OS camera")
        print("     privacy settings, and rerunning.")
        return None

    # ---------------- main loop ----------------
    def run(self):
        print("\n[1] Camera (live)")
        print("[2] Video file")
        print("[3] Auto-generated test video (no camera needed)")
        choice = input("\nSelect: ").strip()

        self.use_audio = input("Enable mock audio pipeline too? (y/N): ").strip().lower() == "y"

        cap = None
        if choice == "1":
            cap = self._open_camera()
            if cap is None:
                return
        elif choice == "2":
            video_path = input("Path to video file: ").strip()
            if not os.path.exists(video_path):
                print(f"File not found: {video_path}")
                return
            cap = cv2.VideoCapture(video_path)
        elif choice == "3":
            test_path = "test_video.mp4"
            if not os.path.exists(test_path):
                self.create_test_video(test_path)
            cap = cv2.VideoCapture(test_path)
        else:
            print("Invalid choice")
            return

        if not cap.isOpened():
            print("Could not open video source!")
            return

        print("Source opened. Press ESC in the video window to quit.\n")
        prev_time = time.time()
        identity_score = 85  # placeholder until ArcFace identity matching is wired in

        is_live_camera = (choice == "1")
        consecutive_failed_reads = 0
        MAX_CONSECUTIVE_FAILS = 30  # ~1s of retries before we give up on a live camera

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    if is_live_camera:
                        # A live camera can drop a frame transiently (backend
                        # hiccup, resolution/format negotiation, USB glitch).
                        # Don't treat a single failed read as "end of stream" --
                        # retry a bit before actually giving up.
                        consecutive_failed_reads += 1
                        if consecutive_failed_reads >= MAX_CONSECUTIVE_FAILS:
                            print("\n[INFO] Camera stopped returning frames")
                            break
                        time.sleep(0.03)
                        continue
                    else:
                        print("\n[INFO] End of stream")
                        break
                consecutive_failed_reads = 0

                frame = cv2.flip(frame, 1)
                self.frame_count += 1
                now = time.time()
                fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 30.0
                prev_time = now

                faces, vision_payload = self.vision_detector.process_frame(frame)
                _, liveness_payload = self.liveness_engine.process_frame(frame, faces)
                deepfake_result = self.deepfake_detector.score_frame(frame)

                mouth_crop = self.liveness_engine.get_mouth_crop(frame, faces)
                if mouth_crop is not None:
                    self.lip_sync.update_mouth_openness(mouth_crop)
                mock_audio = (np.random.rand(1600).astype(np.float32) - 0.5) * 0.2
                self.lip_sync.update_audio(mock_audio)
                sync_result = self.lip_sync.compute_sync_score()

                audio_result = None
                if self.use_audio and self.frame_count % 30 == 0:
                    audio_chunk = np.random.randn(16000).astype(np.float32) * 0.1
                    audio_result = self.audio_detector.ensemble_score(audio_chunk)

                # ---- challenge issuing / display ----
                challenge = liveness_payload.get("challenge_result")
                if challenge:
                    passed = challenge.get("challenge_passed", False)
                    color = (0, 255, 0) if passed else (0, 0, 255)
                    text = f"CHALLENGE: {challenge['challenge_type']} - {'PASS' if passed else 'FAIL'}"
                    cv2.putText(frame, text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if passed:
                        self.challenge_active = False
                        self.challenge_cooldown = self.frame_count + 60
                elif not self.challenge_active and self.frame_count > self.challenge_cooldown:
                    if liveness_payload.get("face_detected") and liveness_payload.get("state") == "IDLE":
                        new_challenge = self.liveness_engine.issue_challenge()
                        self.challenge_active = True
                        print(f"\n>>> NEW CHALLENGE: {new_challenge}")

                if self.challenge_active and liveness_payload.get("state") == "AWAITING_RESPONSE":
                    elapsed = time.time() - self.liveness_engine.challenge_start_time
                    remaining = max(0, int(10 - elapsed))
                    cv2.putText(frame, f"CHALLENGE: {liveness_payload.get('challenge_type', '')} | {remaining}s",
                                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                # ---- fusion ----
                challenge_passed = bool(challenge and challenge.get("challenge_passed"))
                spoof_analysis = vision_payload.get("spoof_analysis") or {}
                raw_spoof_flag = spoof_analysis.get("is_spoof", False) or liveness_payload.get("is_static", False)

                # Temporal smoothing: only treat this as a real spoof once a
                # clear majority of the last N frames agree. A single noisy
                # frame (bad lighting instant, motion blur) shouldn't be
                # able to deny a real user by itself.
                self.spoof_history.append(raw_spoof_flag)
                spoof_ratio = sum(self.spoof_history) / len(self.spoof_history)
                anti_spoof_2d_flag = spoof_ratio >= 0.6 and len(self.spoof_history) >= 5

                scores = {
                    "identity": identity_score,
                    "liveness": (100 if challenge_passed else 50) if not anti_spoof_2d_flag else 20,
                    "visual_deepfake": 100 - self.deepfake_detector.get_visual_deepfake_score(),
                    "audio_spoof": 100 - (self.audio_detector.get_audio_spoof_score() if self.use_audio else 50),
                    "lip_sync": int(sync_result.get("lip_sync_score", 0.5) * 100),
                }
                trust_data = self.compute_trust_score(scores)

                if self.frame_count % 30 == 0:
                    output = self.generate_json(liveness_payload, vision_payload, deepfake_result,
                                                 audio_result, sync_result, trust_data, anti_spoof_2d_flag)
                    print(f"\n--- FUSION OUTPUT ---\n{output}")
                    self.post_to_backend(output)

                # ---- rendering ----
                frame = self.vision_detector.draw_landmarks(frame, faces, vision_payload)

                trust = trust_data["trust_score"]
                band = trust_data["band"]
                band_color = {
                    "TRUSTED": (0, 255, 0),
                    "CAUTION": (0, 255, 255),
                    "SUSPICIOUS": (0, 165, 255),
                    "DENIED": (0, 0, 255),
                }[band]

                bar_x, bar_y, bar_w, bar_h = 20, 180, 200, 20
                filled = int(bar_w * trust / 100)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), band_color, -1)
                cv2.putText(frame, f"TRUST: {trust:.0f} ({band})", (bar_x, bar_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, band_color, 2)
                cv2.putText(frame, f"Weak signal: {trust_data['lowest_signal']}", (bar_x, bar_y + bar_h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                # ---- verdict banner (same hard-deny rule as generate_json) ----
                face_detected = liveness_payload.get("face_detected", False)
                multi_face = liveness_payload.get("multi_face", False)
                if not face_detected:
                    verdict_text, verdict_color = "ACCESS DENIED - NO FACE", (0, 0, 255)
                elif multi_face:
                    verdict_text, verdict_color = "ACCESS DENIED - MULTIPLE FACES", (0, 0, 255)
                elif anti_spoof_2d_flag:
                    verdict_text, verdict_color = "ACCESS DENIED - SPOOF DETECTED", (0, 0, 255)
                elif trust_data["band"] in ("TRUSTED", "CAUTION"):
                    verdict_text, verdict_color = "ACCESS GRANTED", (0, 255, 0)
                else:
                    verdict_text, verdict_color = "ACCESS DENIED", (0, 0, 255)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 4), verdict_color, -1)
                (tw, th), _ = cv2.getTextSize(verdict_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(frame, verdict_text, (frame.shape[1] - tw - 20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, verdict_color, 2)

                cv2.putText(frame, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                df_prob = deepfake_result.get("deepfake_probability", 0.0)
                cv2.putText(frame, f"Deepfake prob: {df_prob:.2f}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"State: {liveness_payload.get('state', 'IDLE')}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Blinks: {liveness_payload.get('blink_count', 0)}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cv2.imshow("DeepfakeGuard - Week 3 Fusion", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("\n[INFO] Quit (ESC)")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.liveness_engine.close()
            self.vision_detector.close()
            print("\n[INFO] System shut down cleanly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepfakeGuard Fusion System")
    parser.add_argument("--no-backend", action="store_true",
                        help="Disable posting fusion data to the Flask backend")
    args = parser.parse_args()
    system = DeepfakeGuardSystem()
    system.no_backend = args.no_backend
    system.run()
