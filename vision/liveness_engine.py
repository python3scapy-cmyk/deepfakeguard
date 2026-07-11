"""
Liveness challenge engine: issues a random challenge (blink twice /
turn head left / turn head right), tracks the response using simple
brightness-based blink detection and face-position-based head yaw,
and reports PASS / FAIL / timeout.

Works purely on OpenCV face boxes (no MediaPipe dependency), so it
runs the same way regardless of Python / MediaPipe version issues.
"""
import random
import time
from enum import Enum

import cv2
import numpy as np


def _has_faces(faces):
    if faces is None:
        return False
    if isinstance(faces, np.ndarray):
        return faces.size > 0
    return len(faces) > 0


class ChallengeState(Enum):
    IDLE = "IDLE"
    CHALLENGE_ISSUED = "CHALLENGE_ISSUED"
    AWAITING_RESPONSE = "AWAITING_RESPONSE"
    PASSED = "PASSED"
    FAILED = "FAILED"


class LivenessChallengeEngine:
    def __init__(self):
        self.state = ChallengeState.IDLE
        self.current_challenge = None
        self.challenge_start_time = 0.0
        self.CHALLENGE_TIMEOUT = 10.0
        self.YAW_THRESHOLD = 15.0

        self.blink_counter = 0
        self.last_blink_time = 0.0
        self.BLINK_COOLDOWN = 0.5
        self.eye_dark_frames = 0
        self.EYE_DARK_CONSECUTIVE = 2

        self.challenges_pool = ["blink_twice", "turn_left", "turn_right"]
        self.completed_challenges = []

        # Anti-spoof: fully static face across many frames looks like a photo
        self.face_position_history = []
        self.STATIC_THRESHOLD = 3
        self.STATIC_FRAMES = 20

    # ---------- anti-spoof helper ----------
    def _check_static_face(self, faces):
        if not _has_faces(faces):
            return False
        x, y, w, h = faces[0]
        center = (x + w // 2, y + h // 2)
        self.face_position_history.append(center)
        if len(self.face_position_history) > self.STATIC_FRAMES:
            self.face_position_history.pop(0)
        if len(self.face_position_history) >= self.STATIC_FRAMES:
            first = self.face_position_history[0]
            for pos in self.face_position_history[1:]:
                if abs(pos[0] - first[0]) > self.STATIC_THRESHOLD or \
                   abs(pos[1] - first[1]) > self.STATIC_THRESHOLD:
                    return False
            return True
        return False

    # ---------- liveness signals ----------
    def _detect_blink(self, frame, faces):
        if not _has_faces(faces):
            return False
        x, y, w, h = faces[0]
        eye_y = y + int(h * 0.25)
        eye_h = int(h * 0.15)
        eye_x = x + int(w * 0.2)
        eye_w = int(w * 0.6)
        eye_roi = frame[eye_y:eye_y + eye_h, eye_x:eye_x + eye_w]
        if eye_roi.size == 0:
            return False
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mean_brightness < 60:
            self.eye_dark_frames += 1
            if self.eye_dark_frames >= self.EYE_DARK_CONSECUTIVE:
                now = time.time()
                if now - self.last_blink_time > self.BLINK_COOLDOWN:
                    self.blink_counter += 1
                    self.last_blink_time = now
                    self.eye_dark_frames = 0
                    return True
        else:
            self.eye_dark_frames = 0
        return False

    def _get_head_yaw(self, frame, faces):
        if not _has_faces(faces):
            return 0.0
        h, w = frame.shape[:2]
        x, y, fw, fh = faces[0]
        center_x = x + fw // 2
        normalized = (center_x - w // 2) / (w // 2)
        return normalized * 30.0

    # ---------- challenge lifecycle ----------
    def issue_challenge(self):
        available = [c for c in self.challenges_pool if c not in self.completed_challenges]
        if not available:
            self.completed_challenges = []
            available = self.challenges_pool[:]
        self.current_challenge = random.choice(available)
        self.state = ChallengeState.CHALLENGE_ISSUED
        self.challenge_start_time = time.time()
        self.blink_counter = 0
        self.eye_dark_frames = 0
        return self.current_challenge

    def process_frame(self, frame, faces=None):
        payload = {
            "state": self.state.value,
            "challenge_type": self.current_challenge,
            "face_detected": False,
            "blink_count": self.blink_counter,
            "yaw": 0.0,
            "challenge_result": None,
            "multi_face": False,
            "is_static": False,
        }

        payload["is_static"] = self._check_static_face(faces)

        if not _has_faces(faces):
            return frame, payload

        payload["face_detected"] = True
        if len(faces) > 1:
            payload["multi_face"] = True
            return frame, payload

        yaw = self._get_head_yaw(frame, faces)
        payload["yaw"] = round(yaw, 1)
        self._detect_blink(frame, faces)
        payload["blink_count"] = self.blink_counter

        if self.state == ChallengeState.CHALLENGE_ISSUED:
            self.state = ChallengeState.AWAITING_RESPONSE

        if self.state == ChallengeState.AWAITING_RESPONSE:
            elapsed = time.time() - self.challenge_start_time
            if elapsed > self.CHALLENGE_TIMEOUT:
                self.state = ChallengeState.FAILED
                payload["challenge_result"] = {
                    "challenge_type": self.current_challenge,
                    "challenge_passed": False,
                    "reason": "Timeout: no response within 10 seconds",
                    "fail_reason": "timeout",
                }
            else:
                passed, reason = False, ""
                if self.current_challenge == "blink_twice" and self.blink_counter >= 2:
                    passed, reason = True, "Two blinks detected"
                elif self.current_challenge == "turn_left" and yaw < -self.YAW_THRESHOLD:
                    passed, reason = True, f"Head turned left ({yaw:.1f} deg)"
                elif self.current_challenge == "turn_right" and yaw > self.YAW_THRESHOLD:
                    passed, reason = True, f"Head turned right ({yaw:.1f} deg)"

                if passed:
                    response_time = (time.time() - self.challenge_start_time) * 1000
                    self.state = ChallengeState.PASSED
                    self.completed_challenges.append(self.current_challenge)
                    payload["challenge_result"] = {
                        "challenge_type": self.current_challenge,
                        "challenge_passed": True,
                        "blinks_detected": self.blink_counter,
                        "response_time_ms": round(response_time, 1),
                        "reason": reason,
                    }

        return frame, payload

    def get_mouth_crop(self, frame, faces):
        if not _has_faces(faces):
            return None
        h, w = frame.shape[:2]
        x, y, fw, fh = faces[0]
        mouth_y = max(0, y + int(fh * 0.65))
        mouth_h = min(int(fh * 0.3), h - mouth_y)
        mouth_x = max(0, x + int(fw * 0.2))
        mouth_w = min(int(fw * 0.6), w - mouth_x)
        if mouth_h <= 0 or mouth_w <= 0:
            return None
        crop = frame[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (96, 96))

    def reset(self):
        self.state = ChallengeState.IDLE
        self.current_challenge = None
        self.blink_counter = 0
        self.eye_dark_frames = 0
        self.completed_challenges = []
        self.face_position_history = []

    def close(self):
        pass
