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
from collections import deque
from enum import Enum

import os

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
        # 6s: long enough to read the prompt and act, short enough that a
        # failed attempt doesn't stall the whole verification. Users
        # consistently respond within ~2-3s once they see the prompt.
        self.CHALLENGE_TIMEOUT = 6.0
        self.YAW_THRESHOLD = 15.0        # legacy absolute threshold (kept for payload compat)
        # Turn detection is now BASELINE-RELATIVE with PEAK tracking:
        #  - baseline = face position when the challenge was issued, so a
        #    user standing off-center isn't pre-biased toward one side;
        #  - peak = strongest excursion seen during the challenge, so the
        #    moment of deepest turn counts even if MediaPipe then loses the
        #    face at full profile (the exact failure users hit on
        #    turn_right) -- the pass is evaluated on the peak, not on
        #    whatever the yaw happens to be on the current frame.
        self.REL_YAW_THRESHOLD = 8.0    # translation path: clear sideways head movement
        # ROTATION path: turning the head IN PLACE barely moves the Haar box
        # center (that proxy measures translation), but it reliably NARROWS
        # the box - a profile is thinner than a frontal face. Rotation is
        # therefore detected as: box width shrunk below WIDTH_SHRINK_RATIO
        # of its challenge-issue baseline, WITH the center drifting in the
        # challenge direction by at least ROT_MIN_DRIFT (sign disambiguates
        # left vs right; magnitude requirement is small on purpose).
        self.WIDTH_SHRINK_RATIO = 0.85
        self.ROT_MIN_DRIFT = 2.0
        # PROFILE detection - the definitive rotation signal. The frontal
        # Haar cascade (which feeds `faces`) doesn't gradually narrow its
        # box as the head turns; it simply STOPS detecting at ~25-30 deg,
        # so width-shrink evidence often never materializes (confirmed
        # frame-by-frame on a user recording). OpenCV ships a profile
        # cascade: it matches faces pointing toward image-LEFT; running it
        # on the mirrored copy matches image-RIGHT. In this pipeline the
        # analysis stream is mirrored (main.py cv2.flip convention), where
        # turning YOUR head right points the nose toward image-right.
        # If left/right ever verify swapped on some capture setup, flip
        # this one constant:
        self.PROFILE_MAPPING_FLIPPED = False
        # TRUE yaw path: when the caller can supply a real 3D head yaw
        # (engine.py passes InsightFace's pose during turn challenges),
        # turn evidence comes straight from it - no proxies. Flip the sign
        # constant if left/right verify swapped on your capture setup
        # (watch the [LIVENESS] true_yaw log line to confirm).
        self.TRUE_YAW_THRESHOLD = 16.0   # degrees
        self.TRUE_YAW_FLIPPED = False
        self._last_true_yaw_log = 0.0
        prof_path = os.path.join(getattr(cv2.data, "haarcascades", ""),
                                 "haarcascade_profileface.xml")
        self._profile_cascade = cv2.CascadeClassifier(prof_path)
        if self._profile_cascade.empty():
            print("[LIVENESS][WARN] profile cascade not found - turn "
                  "challenges fall back to drift/width evidence only")
        self._last_profile_log = 0.0
        self._yaw_baseline = None
        self._width_baseline = None
        self._peak_rel_yaw = 0.0
        self._turn_evidence = {"turn_left": False, "turn_right": False}

        self.blink_counter = 0
        self.last_blink_time = 0.0
        self.BLINK_COOLDOWN = 0.35
        self.eye_dark_frames = 0
        self.EYE_DARK_CONSECUTIVE = 2
        # REAL blink detection via Eye Aspect Ratio supplied by the caller
        # (participant.html computes it from the MediaPipe mesh it already
        # runs; main.py can pass None and keeps the legacy brightness path).
        # The threshold is ADAPTIVE: eye shape varies hugely per person and
        # per camera distance, so a fixed EAR cutoff either never fires or
        # fires constantly. Baseline = a high percentile of recent open-eye
        # EARs; a blink is a dip well below that, followed by recovery.
        self.EAR_DROP_RATIO = 0.72     # dip below 72% of baseline = eye closing
        self.EAR_RECOVER_RATIO = 0.85  # back above 85% = eye reopened
        self._ear_history = deque(maxlen=90)
        self._ear_closed = False

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
    def _update_blink_from_ear(self, ear):
        """State machine on the real EAR: open -> closed -> open counts one
        blink. Returns True on the frame a blink completes."""
        if ear is None or ear <= 0:
            return False
        self._ear_history.append(float(ear))
        if len(self._ear_history) < 8:
            return False   # need a baseline first

        baseline = float(np.percentile(self._ear_history, 80))
        if baseline <= 0:
            return False
        ratio = ear / baseline

        if not self._ear_closed:
            if ratio < self.EAR_DROP_RATIO:
                self._ear_closed = True
            return False

        # currently closed - wait for reopen
        if ratio > self.EAR_RECOVER_RATIO:
            self._ear_closed = False
            now = time.time()
            if now - self.last_blink_time > self.BLINK_COOLDOWN:
                self.blink_counter += 1
                self.last_blink_time = now
                print(f"[LIVENESS] blink #{self.blink_counter} "
                      f"(ear={ear:.3f} baseline={baseline:.3f})")
                return True
        return False

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
        self._ear_closed = False
        self._yaw_baseline = None
        self._width_baseline = None
        self._peak_rel_yaw = 0.0
        self._turn_evidence = {"turn_left": False, "turn_right": False}
        return self.current_challenge

    def process_frame(self, frame, faces=None, true_yaw=None, ear=None):
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
            # Losing the face mid-turn is EXPECTED at full profile. Keep
            # evaluating the challenge on the peak yaw recorded so far
            # instead of freezing until the face comes back.
            if self.state == ChallengeState.AWAITING_RESPONSE:
                if self.current_challenge in ("turn_left", "turn_right"):
                    d = self._detect_profile_direction(frame)
                    if d:
                        self._turn_evidence[d] = True
                payload["challenge_result"] = self._evaluate_turn_peak_or_timeout()
                payload["state"] = self.state.value
            return frame, payload

        payload["face_detected"] = True
        if len(faces) > 1:
            payload["multi_face"] = True
            return frame, payload

        yaw = self._get_head_yaw(frame, faces)
        if self.state in (ChallengeState.CHALLENGE_ISSUED, ChallengeState.AWAITING_RESPONSE):
            if self._yaw_baseline is None:
                self._yaw_baseline = yaw
                self._width_baseline = float(faces[0][2])
            rel_yaw = yaw - self._yaw_baseline
            if abs(rel_yaw) > abs(self._peak_rel_yaw):
                self._peak_rel_yaw = rel_yaw
            # Rotation evidence: narrowed box + drift sign in the turn direction
            if self._width_baseline:
                width_ratio = float(faces[0][2]) / self._width_baseline
                if width_ratio <= self.WIDTH_SHRINK_RATIO:
                    if rel_yaw <= -self.ROT_MIN_DRIFT:
                        self._turn_evidence["turn_left"] = True
                    if rel_yaw >= self.ROT_MIN_DRIFT:
                        self._turn_evidence["turn_right"] = True
            # Translation evidence (clear sideways movement) also counts
            if self._peak_rel_yaw < -self.REL_YAW_THRESHOLD:
                self._turn_evidence["turn_left"] = True
            if self._peak_rel_yaw > self.REL_YAW_THRESHOLD:
                self._turn_evidence["turn_right"] = True
            # REAL 3D yaw (definitive when provided)
            if true_yaw is not None:
                ty = -true_yaw if self.TRUE_YAW_FLIPPED else true_yaw
                now_ = time.time()
                if now_ - self._last_true_yaw_log > 0.7:
                    print(f"[LIVENESS] true_yaw={ty:+.1f} deg "
                          f"(challenge={self.current_challenge})")
                    self._last_true_yaw_log = now_
                if ty >= self.TRUE_YAW_THRESHOLD:
                    self._turn_evidence["turn_right"] = True
                if ty <= -self.TRUE_YAW_THRESHOLD:
                    self._turn_evidence["turn_left"] = True
        else:
            rel_yaw = 0.0
        payload["yaw"] = round(yaw, 1)
        payload["rel_yaw"] = round(rel_yaw, 1)
        if ear is not None:
            self._update_blink_from_ear(ear)      # real EAR path (browser mesh)
        else:
            self._detect_blink(frame, faces)      # legacy brightness fallback
        payload["blink_count"] = self.blink_counter
        payload["ear"] = round(ear, 3) if ear is not None else None

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
                elif (self.current_challenge in ("turn_left", "turn_right")
                        and self._turn_evidence[self.current_challenge]):
                    passed = True
                    reason = (f"Head turn detected ({self.current_challenge}, "
                              f"peak drift {self._peak_rel_yaw:.1f})")

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

    def _detect_profile_direction(self, frame):
        """Run the profile cascade on the frame and its mirror. Returns
        'turn_left' / 'turn_right' / None. Called only on frames where the
        frontal cascade lost the face during an active turn challenge, so
        the cost (~5-10ms at 320px) is paid exactly when it matters."""
        if self._profile_cascade.empty() or frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 320.0 / gray.shape[1]
        gray = cv2.resize(gray, (320, max(1, int(gray.shape[0] * scale))))
        gray = cv2.equalizeHist(gray)
        hits_imgleft = self._profile_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=4, minSize=(40, 40))
        hits_imgright = self._profile_cascade.detectMultiScale(
            cv2.flip(gray, 1), scaleFactor=1.15, minNeighbors=4, minSize=(40, 40))

        def _area(hits):
            return max((w * h for (_x, _y, w, h) in hits), default=0)

        a_left, a_right = _area(hits_imgleft), _area(hits_imgright)
        if a_left == 0 and a_right == 0:
            return None
        # Mirrored stream: nose toward image-right == user turned right.
        direction = "turn_right" if a_right >= a_left else "turn_left"
        if self.PROFILE_MAPPING_FLIPPED:
            direction = ("turn_left" if direction == "turn_right"
                         else "turn_right")
        now = time.time()
        if now - self._last_profile_log > 1.0:
            print(f"[LIVENESS] profile face detected -> {direction} "
                  f"(areas L={a_left} R={a_right})")
            self._last_profile_log = now
        return direction

    def _evaluate_turn_peak_or_timeout(self):
        """Called while the face is LOST during AWAITING_RESPONSE. If the
        recorded peak already satisfies the turn, pass it (turning far
        enough to defeat the face detector is the strongest possible
        'I turned my head' evidence, not a failure). On timeout, fail."""
        if (self.current_challenge in ("turn_left", "turn_right")
                and self._turn_evidence.get(self.current_challenge)):
            self.state = ChallengeState.PASSED
            self.completed_challenges.append(self.current_challenge)
            return {"challenge_type": self.current_challenge, "challenge_passed": True,
                    "reason": (f"Head turn detected ({self.current_challenge}, "
                               f"face lost at profile - strongest rotation evidence)"),
                    "response_time_ms": round((time.time() - self.challenge_start_time) * 1000, 1)}
        # Losing the face DURING a turn challenge, shortly after issue, with
        # a narrowed last-seen box is itself rotation-consistent; but without
        # direction evidence we can't credit a side - fall through to timeout.
        if time.time() - self.challenge_start_time > self.CHALLENGE_TIMEOUT:
            self.state = ChallengeState.FAILED
            return {"challenge_type": self.current_challenge, "challenge_passed": False,
                    "reason": "Timeout (face not visible)", "fail_reason": "timeout"}
        return None

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
